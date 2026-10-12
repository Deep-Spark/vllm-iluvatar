def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors: IntermediateTensors | None, inputs_embeds: torch.Tensor | None=None, lookback_token_ids: torch.Tensor | None=None) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    """Embed -> local layers -> leftover FFN post -> collapse or PP send.

        Returns ``(hidden_states, aux)`` instead of a bare tensor when a DSpark
        draft asked for aux layers. Aux captured on an earlier PP rank is packed
        into the intermediate tensors (``EagleModelMixin``) and concatenated on
        the last rank, which is where the drafter runs. PP=1 leaves the packed
        keys empty and returns only this rank's captures.
        """
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors['hidden_states']
        pre_mix = intermediate_tensors['pre_mix']
        self._accept_remote_kv(intermediate_tensors, positions)
        self._accept_remote_candidates(intermediate_tensors)
    self._prepare_engram(input_ids, positions, lookback_token_ids)
    residual = post_mix = res_mix = None
    aux_hidden_states: list[torch.Tensor] = []
    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer), self.start_layer):
        hidden_states, residual, post_mix, res_mix, pre_mix = layer(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)
        if idx + 1 in self.aux_hidden_state_layers:
            aux_hidden_states.append(mhc.hc_post(hidden_states, residual, post_mix, res_mix).mean(dim=1))
    hidden_states = mhc.hc_post(hidden_states, residual, post_mix, res_mix)
    _p21_capture(self, hidden_states, pre_mix, positions)
    if not get_pp_group().is_last_rank:
        tensors = {'hidden_states': hidden_states, 'pre_mix': pre_mix}
        num_tokens = positions.shape[0]
        if self.pp_plan.sends_kv:
            assert self.main_kv_rows is not None
            assert self.indexer_kv_rows is not None
            tensors['main_kv'] = self.main_kv_rows[:num_tokens]
            tensors['indexer_kv'] = self.indexer_kv_rows[:num_tokens]
        if self.pp_plan.sends_topk:
            tensors['topk'] = self.topk_indices_buffer[:num_tokens]
        if self.pp_plan.sends_candidates:
            assert self.candidate_idxs_buffer is not None
            tensors['candidates'] = self.candidate_idxs_buffer[:num_tokens]
        tensors.update(self.pack_local_aux_hidden_states(aux_hidden_states))
        return IntermediateTensors(tensors)
    hidden_states = mhc.hc_pre(hidden_states, pre_mix)
    hidden_states = self.norm(hidden_states)
    aux_hidden_states = self.collect_remote_aux_hidden_states(intermediate_tensors) + aux_hidden_states
    if aux_hidden_states:
        return (hidden_states, aux_hidden_states)
    return hidden_states
