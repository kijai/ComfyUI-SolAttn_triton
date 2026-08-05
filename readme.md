## Experimental Sol-Attn implementation for ComfyUI

work in progress

https://owen718.github.io/pubs/43-2026-arxiv-solattn/

Only tested on 4090 and 5090 with MiniMax H3. Uses triton so first run will be slower.

Balance the quality/speed with start/end percent and tau.

Tests:

https://github.com/user-attachments/assets/8d9ed820-0417-4d68-9d1c-5199534bed3b


<table>
<tr>
<td align="center"><b>Sageattn</b></td>
<td align="center"><b>Sol-attn</b></td>
</tr>
<tr>
<td width="50%">
<video src="https://github.com/user-attachments/assets/27f201ea-6bfc-4f43-826c-51809eed9d15" controls muted loop></video>
</td>
<td width="50%">
<video src="https://github.com/user-attachments/assets/73f63d14-2166-4f62-b098-e817ec1d7704" controls muted loop></video>
</td>
</tr>
</table>


<img width="482" height="500" alt="image" src="https://github.com/user-attachments/assets/27ae9886-aa3e-4470-a507-3a7c52b24be5" />
