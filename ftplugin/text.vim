function! s:MapLamaKeys() abort
  if get(g:, 'lama_no_default_mappings', 0)
    return
  endif
  imap <C-l> <Plug>(lama-toggle)
  nmap <C-l> <Plug>(lama-toggle)

  imap <silent><script><nowait><expr> <C-]> lama#Stop() . "\<C-]>"
  " Not mapping C-] in normal mode because it's already mapped to jumping
endfunction

call s:MapLamaKeys()
