# utils.py
import json
import os
import time
import requests
from typing import Dict, Any, List, Optional
import subprocess
from openai import OpenAI


def filter_and_fix_file(file_path):
    """
    Reads a JSONL file, removes invalid lines, and overwrites the original file with only valid lines.
    """
    valid_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.strip():  # Check if the line is not empty
                try:
                    json.loads(line)  # Attempt to load the line as JSON
                    valid_lines.append(line)  # Store valid lines
                except json.JSONDecodeError:
                    print(f"Invalid JSON line removed: {line.strip()}")  # Log invalid line
    
    # Overwrite the original file with valid lines
    with open(file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(valid_lines)

def chat_completion(api_base: str, model_name: str, messages: list, max_tokens: int = 256, temperature: float = 0.7, api_key: str = "xxx") -> str:
    """Generic helper for chat completion supporting both vLLM and API modes."""
    if '/v1' not in api_base:
        api_base = api_base + '/v1'
    
    client = OpenAI(base_url=api_base, api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return completion.choices[0].message.content

def read_jsonl(file_path):
    """
    Read JSONL file safely without modifying it.
    Skips invalid lines but does not overwrite the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARNING] Skipping invalid JSON line in {file_path}: {line.strip()[:100]}")
                    continue

def write_jsonl(file_path, data_list, append=False):
    """
    Writes a list of dictionaries to a JSONL file.
    If append is True, appends the data to the file instead of overwriting it.
    """
    mode = 'a' if append else 'w'
    
    # check if file exists
    if not os.path.exists(file_path):
        # check the parent directory and create if it doesn't exist
        parent_dir = os.path.dirname(file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        
        # create the file if it doesn't exist
        with open(file_path, 'w', encoding='utf-8') as f:
            pass
        
        # update mode to write
        mode = 'w'


    with open(file_path, mode, encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def start_vllm_server(model_path: str, model_name: str, port: int, gpu: int = 1, timeout: int = 1800):
    """
    Launches a vLLM OpenAI API server via subprocess.
    model_path: The path or name of the model you want to host
    port: Which port to host on
    gpu: The tensor-parallel-size (number of GPUs)
    timeout: Maximum time to wait for server startup (seconds)
    
    Returns:
        process: The subprocess.Popen object if successful, None if failed
    """
    # Command to activate conda environment and start the server
    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={model_path}',
        f"--served-model-name={model_name}",
        f'--tensor-parallel-size={gpu}',
        f"--gpu-memory-utilization=0.85",
        f'--port={port}',
        '--trust-remote-code'
    ]

    try:
        process = subprocess.Popen(command, shell=False)
        
        # Check if process died immediately (within 5 seconds)
        time.sleep(5)
        if process.poll() is not None:
            print(f"[ERROR] vLLM server process died immediately with return code {process.returncode}")
            return None
        
        # Wait for server to be ready, checking process health periodically
        server_ready = wait_for_server_with_process_check(
            f"http://localhost:{port}", 
            process, 
            timeout
        )
        
        if not server_ready:
            print(f"[ERROR] vLLM server failed to start or died during startup")
            try:
                process.terminate()
                process.wait(timeout=10)
            except:
                process.kill()
            return None
        
        print(f"[INFO] Started vLLM server for model '{model_path}' on port {port} (GPU={gpu}).")
        return process
        
    except Exception as e:
        print(f"[ERROR] Failed to start vLLM server: {e}")
        return None


def start_vllm_server_with_gpus(model_path: str, model_name: str, port: int, gpus: List[int]):
    """
    Launches a vLLM OpenAI API server via subprocess with specific GPUs assigned.

    Parameters:
    model_path: str - The path or name of the model you want to host.
    model_name: str - The name of the model to be served.
    port: int - The port to host the server on.
    gpus: List[int] - List of GPU indices to be assigned for this server.

    Returns:
    process: subprocess.Popen - The process running the vLLM server.
    """
    gpu_list = ",".join(map(str, gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={model_path}',
        f'--served-model-name={model_name}',
        f'--tensor-parallel-size={len(gpus)}',
        '--gpu-memory-utilization=0.85',
        f'--port={port}',
        '--trust-remote-code'
    ]

    process = subprocess.Popen(command, shell=False, env=os.environ.copy())
    
    wait_for_server(f"http://localhost:{port}", 600)

    print(f"[INFO] Started vLLM server for model '{model_name}' on port {port} with GPUs {gpu_list}.")

    return process

def allocate_gpus(total_gpus: int, processes: int) -> List[List[int]]:
    """
    Allocate GPUs for multiple processes.

    Parameters:
    total_gpus: int - Total number of GPUs available.
    processes: int - Number of processes to allocate GPUs for.

    Returns:
    List[List[int]] - A list where each sublist contains the GPUs assigned to a process.
    """
    if total_gpus < processes:
        raise ValueError("Not enough GPUs available for the number of processes.")

    gpus_per_process = total_gpus // processes
    extra_gpus = total_gpus % processes

    allocation = []
    start = 0

    for i in range(processes):
        end = start + gpus_per_process + (1 if i < extra_gpus else 0)
        allocation.append(list(range(start, end)))
        start = end

    return allocation



def wait_for_server(url: str, timeout: int = 600):
    """
    Polls the server's /models endpoint until it responds with HTTP 200 or times out.
    """
    start_time = time.time()
    while True:
        try:
            r = requests.get(url + "/v1/models", timeout=3)
            if r.status_code == 200:
                print("[INFO] vLLM server is up and running.")
                return True
        except Exception:
            pass
        if time.time() - start_time > timeout:
            print(f"[ERROR] Server did not start at {url} within {timeout} seconds.")
            return False
        time.sleep(2)


def wait_for_server_with_process_check(url: str, process, timeout: int = 600):
    """
    Polls the server's /models endpoint until it responds with HTTP 200 or times out.
    Also checks if the subprocess is still alive during waiting.
    
    Returns:
        True if server is ready, False if timeout or process died
    """
    start_time = time.time()
    check_interval = 2
    
    while True:
        # Check if process is still alive
        if process.poll() is not None:
            print(f"[ERROR] vLLM server process died with return code {process.returncode}")
            return False
        
        try:
            r = requests.get(url + "/v1/models", timeout=3)
            if r.status_code == 200:
                print("[INFO] vLLM server is up and running.")
                return True
        except Exception:
            pass
        
        if time.time() - start_time > timeout:
            print(f"[ERROR] Server did not start at {url} within {timeout} seconds.")
            return False
        
        time.sleep(check_interval)
        
def stop_vllm_server(process):
    process.terminate()
    process.wait()
    print("[INFO] Stopped vLLM server.")

