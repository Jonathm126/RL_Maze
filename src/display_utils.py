import IPython
import base64
import subprocess
import webbrowser
from torch.utils.tensorboard import SummaryWriter


def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''-
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

def launch_tb(log_dir, port = 6008):
  tb_writer = SummaryWriter(log_dir)
  print(f"TensorBoard logs are saved in: {log_dir}")
  
  # launch tb
  tb_process = subprocess.Popen(["tensorboard", f"--logdir={log_dir}", f"--port={port}", "--host=localhost"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  webbrowser.open(f"http://localhost:{port}");
  return tb_process, tb_writer

