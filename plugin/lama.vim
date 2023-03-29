if exists('g:loaded_lama')
  finish
endif
let g:loaded_lama = 1

scriptencoding utf-8

function! s:ColorScheme() abort
  if &t_Co == 256
    hi def LamaSuggestion guifg=#b0aaff ctermfg=244
  else
    hi def LamaSuggestion guifg=#b0aaff ctermfg=8
  endif
  hi def link LamaAnnotation Normal
endfunction

call s:ColorScheme()

imap <Plug>(lama-stop)     <Cmd>call lama#Stop()<CR>
nmap <Plug>(lama-stop)     <Cmd>call lama#Stop()<CR>

imap <Plug>(lama-toggle)     <Cmd>call lama#Toggle()<CR>
nmap <Plug>(lama-toggle)     <Cmd>call lama#Toggle()<CR>

nmap <Plug>(lama-install)     <Cmd>call lama#Install()<CR>

