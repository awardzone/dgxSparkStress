'''
Note tecniche e personalizzazione:


1) Tensor Cores: L'uso di torch.float16 (Half Precision) è fatto appositamente 
per attivare i Tensor Cores, che sono i componenti della GPU che consumano più 
energia e generano più calore in assoluto sulle DGX.

2) Saturazione VRAM: Se la DGX ha schede con molta memoria (es. 40GB, 80GB o più 
per GPU), potresti voler aumentare il valore di BATCH_SIZE (es. BATCH_SIZE = 12) 
o MATRIX_SIZE (es. 24576) finché l'utilizzo della memoria mostrato da nvidia-smi 
non sfiora il 90-95%. Attenzione a non esagerare per evitare l'errore di Out of 
Memory (OOM).

3) Multiprocessing: Lo script usa il modulo multiprocessing di Python. 
Questo è fondamentale sulle DGX per bypassare il GIL (Global Interpreter Lock) 
di Python e garantire che ogni GPU riceva comandi alla massima velocità possibile 
da un core CPU dedicato.


Calcolo dell'impatto sulla VRAM:
- Una matrice di dimensioni 32768 * 32768 contiene circa 1,07 miliardi di elementi.
- In precisione torch.float16 (FP16), ogni elemento occupa 2 byte. 
Quindi, una singola matrice occupa circa 2,15 GB di VRAM.
- Nel ciclo allochiamo matrices_A e matrices_B, quindi 24 + 24 = 48 matrici 
in totale per GPU.
- 48 matrici *  2,15GB approx 103,2GB di VRAM occupata solo per i dati di partenza.
- Il resto della VRAM (fino a circa 115-120 GB) verrà dinamicamente saturato dai 
tensori temporanei generati dalle operazioni di moltiplicazione (torch.matmul) 
e dai contesti CUDA.
'''

import torch
import time
import sys
from multiprocessing import Process
import torch.multiprocessing as mp  # Usiamo il multiprocessing specifico di PyTorch

#MATRIX_SIZE = 16384  # Dimensione della matrice (16k x 16k). Aumenta se la VRAM è enorme (es. H100/A100)
#BATCH_SIZE = 4       # Numero di matrici allocate contemporaneamente per saturare la VRAM
#ITERATIONS = 100000  # Numero di iterazioni per processo

# Configurazione ottimizzata per GPU con 128 GB di VRAM
MATRIX_SIZE = 32768  # Matrici gigantesche da 32k x 32k
BATCH_SIZE = 24      # Numero di matrici allocate in parallelo per saturare la VRAM
ITERATIONS = 200000  # Ciclo lungo per mantenere lo stress stabile
tensor_type = torch.float16 

def stress_gpu(gpu_id):
    """Funzione eseguita su una singola GPU per metterla sotto stress."""
    try:
        # Assegna il processo alla GPU specifica
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        print(f"[GPU {gpu_id}] Avvio stress test su {torch.cuda.get_device_name(gpu_id)}...")
        sys.stdout.flush()
        
        print(f"[GPU {gpu_id}] Allocazione memoria in corso (Matrix Size: {MATRIX_SIZE}x{MATRIX_SIZE})...")
        # Alloca le matrici per riempire i 128GB di VRAM
        matrices_A = [torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=tensor_type, device=device) for _ in range(BATCH_SIZE)]
        matrices_B = [torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=tensor_type, device=device) for _ in range(BATCH_SIZE)]
        
        print(f"[GPU {gpu_id}] Memoria allocata con successo. Inizio ciclo di calcolo pesante...")
        sys.stdout.flush()

        start_time = time.time()
        
        for i in range(ITERATIONS):
            # Esegue moltiplicazioni di matrici a catena (sfrutta al 100% i Tensor Cores)
            for j in range(BATCH_SIZE):
                _ = torch.matmul(matrices_A[j], matrices_B[j])
            
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"[GPU {gpu_id}] Iterazione {i}/{ITERATIONS} completata. Tempo parziale: {elapsed:.2f}s")
                sys.stdout.flush()

    except Exception as e:
        print(f"[GPU {gpu_id}] Errore durante l'esecuzione: {e}")
    finally:
        print(f"[GPU {gpu_id}] Test terminato.")

if __name__ == '__main__':
    # !!! COSA FONDAMENTALE PER RISOLVERE L'ERRORE !!!
    # Forza Python a usare 'spawn' invece di 'fork', ripulendo i contesti CUDA per ogni processo
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Già impostato

    # Verifica la disponibilità di CUDA
    if not torch.cuda.is_available():
        print("Errore: CUDA non è disponibile. Questo script richiede GPU NVIDIA.")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print(f"=== NVIDIA DGX STRESS TEST ===")
    print(f"Rilevate {num_gpus} GPU disponibili.")
    print("ATTENZIONE: Questo script spingerà il consumo energetico e le temperature al massimo.")
    print("==============================")
    
    processes = []
    
    # Avvia un processo separato per ogni GPU
    for gpu_id in range(num_gpus):
        p = Process(target=stress_gpu, args=(gpu_id,))
        processes.append(p)
        p.start()

    # Attendi il completamento di tutti i processi
    for p in processes:
        p.join()

    print("Stress test completato.")
