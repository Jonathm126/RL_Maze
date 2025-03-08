import IPython
import base64
import subprocess
import webbrowser
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def plot_metrics(metrics, model_name="My Model"):
    # Create a figure with 1 row and 3 columns of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Add a rotated title on the left side of the figure using fig.text.
    # Adjust the x, y coordinates as needed (here 0.02 for x near the left and 0.5 for y centered).
    fig.text(0.02, 0.5, model_name, rotation=90, fontsize=16, ha='center', va='center')
    
    # Left subplot: Display average metrics text
    axes[0].axis('off')  # Hide the axis
    textstr = '\n'.join((
        f"Average Reward: {metrics['avg_reward']:.2f}",
        f"Success Rate: {metrics['success_rate']:.2%}",
        f"Avg Steps to Done: {metrics['avg_steps_to_done']:.2f}"
    ))
    axes[0].text(0.5, 0.5, textstr, horizontalalignment='center', verticalalignment='center',
                 fontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
    
    # Middle subplot: Plot rewards per episode
    axes[1].plot(metrics["all_rewards"], marker='o')
    axes[1].set_title("Episode Rewards")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    
    # Right subplot: Plot steps per episode
    axes[2].plot(metrics["all_steps"], marker='o', color='orange')
    axes[2].set_title("Episode Steps")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Steps")
    
    plt.tight_layout()
    plt.show()



