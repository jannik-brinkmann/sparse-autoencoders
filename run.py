import os
import multiprocessing
from dataclasses import replace
from datetime import datetime
from multiprocessing import Semaphore

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import CachedActivationLoader, Trainer, TrainingConfig

multiprocessing.set_start_method('spawn', force=True)

config = TrainingConfig(
        # Base Model
        model_name_or_path = "EleutherAI/pythia-70m-deduped",
        dataset_name_or_path = "Elriggs/openwebtext-100k",
        hook_point = "gpt_neox.layers.3",
        lr_warmup_steps = 100, 
        expansion_factor = 4,
        steps = 3440,
        use_ghost_grads = False,
        sparsity_coefficient = 3e-3,
        
        # Weights and Biases
        use_wandb = True,
        wandb_entity = "jannik-brinkmann",
        wandb_project = "sparse-autoencoder",
    )
configs = [config]


def worker_process(queue, config, semaphore):
    """
    :param queue: multiprocessing.Queue instance for receiving activations.
    :param config: Configuration to initialize the Trainer.
    """
    
    # update unique identifier of the process
    pid = os.getpid()
    config = replace(config, pid=str(pid))
    
    trainer = Trainer(config)
    
    while True:
        
        # get activations from the main process
        semaphore.acquire()
        activations = queue.get()
        semaphore.release()
        if activations is None:
            semaphore.release()
            break
        
        # train the model on these activations
        trainer.step(activations.to(config.device))
    
    # save the trained model
    trainer.save_weights()


def training(configs):
    
    # generate a UUID for the training run
    run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    configs = [replace(c, run_id=run_id) for c in configs]
    run_dir = os.path.join("outputs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # determine activation size
    activation_loader = CachedActivationLoader(config)
    activations = activation_loader.get(0, split="train")
    activation_size = activations.size(-1)
    configs = [replace(c, activation_size=activation_size) for c in configs]
    
    # determine the number of autoencoders that should be trained in parallel
    n_autoencoders = len(configs)
    
    # create separate queues for sending activations to each worker processes
    queues = [multiprocessing.Queue() for _ in range(n_autoencoders)]
    
    # setup a limit for the queue size to avoid out-of-memory issues
    queue_size_limit = 10
    semaphore = Semaphore(queue_size_limit)
    
    # create and start worker processes
    workers = []
    for i in range(n_autoencoders):
        worker = multiprocessing.Process(
            target=worker_process, 
            args=(queues[i], configs[i], semaphore)
        )
        worker.start()
        workers.append(worker)
        
    # generate activations and send them to the worker processes
    for i in range(3440): #TODO: change number to config.steps, right?
        activations = activation_loader.get(i, split="train")
        for q in queues:
            semaphore.acquire()
            q.put(activations)
            semaphore.release()
    
    # tell workers to stop
    for q in queues:
        q.put(None)
        
    # wait for workers to finish
    for w in workers:
        w.join()  
        
        
if __name__ == "__main__":
    training(configs)
    