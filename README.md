Script Python per testare le macchine Nvidia DGX Spark

dgxStress.py:
-------------
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

************************************************************************************
dgxStressLLM.py
---------------
Per caricare un modello che saturi una GPU da 128 GB di VRAM senza dover scaricare 
centinaia di gigabyte di file da internet (che richiederebbero ore), usiamo un 
trucco avanzato di PyTorch e Hugging Face: inizializziamo una configurazione di 
un modello colossale (il formato Llama-3-70B) usando il contesto meta di PyTorch 
(che crea l'architettura a costo zero in RAM), e poi allochiamo i tensori reali 
direttamente sulla GPU convertendoli in FP16.
Per stressare al massimo la macchina, lo script non fa solo calcoli casuali, 
ma esegue continui cicli di Forward Pass (generazione/inferenza) con sequenze 
a contesto massimo.


1) mp.set_start_method('spawn', force=True) all'inizio assoluto: Risolve 
definitivamente l'errore del sub-processo forkato, isolando la memoria CUDA di 
ogni processo figlio.

2) Uso di with torch.device(device):: Inizializza l'enorme ammontare di parametri 
direttamente dentro la VRAM della GPU selezionata. Se non usassimo questo 
accorgimento, PyTorch creerebbe prima il modello nella RAM di sistema (CPU), 
saturando immediatamente i nodi NUMA della DGX e causando un crash di sistema 
per Out-Of-Memory della CPU (OOM killer di Linux).

3) Saturazione di Tipo "Transformer": A differenza della moltiplicazione di matrici 
standard, l'architettura a blocchi di Llama costringe la GPU a muovere 
continuamente i pesi del modello dai chip di memoria HBM3 ai core di calcolo 
e viceversa, generando uno stress combinato su ampiezza di banda della memoria 
(Memory Bandwidth) e potenza di calcolo puro (TFLOPS).

************************************************************************************

NOTA:

Lo script va bene anche per testare le H200 con architettura Hopper, va considerato però che queste sono un 'tantino' :p 
più performanti quindi si possono introdurre piccole modifiche per usarle a pieno, infatti H200 NVL hanno un Power Cap di 600W
ciascuna e circa 141 GB  di VRAM utilizzabile.

-------------
1) Ottimizzazione della VRAM per le H200

Lo script originale calcola circa 103,2 GB di allocazione fissa, pensata per schede da 128 GB. Le H200 hanno poco più di 140 GB.
Se lanciando lo script noti tramite nvidia-smi che l'utilizzo della memoria si ferma intorno all'80-85%, puoi spingerlo oltre 
aumentando leggermente il BATCH_SIZE:

    BATCH_SIZE = 28 (oppure 30)
    
Se ricevi un errore di CUDA Out of Memory (OOM) dovuto ai tensori temporanei creati durante il torch.matmul, scendi gradualmente (es. 26).

-------------
2) Passaggio a BFloat16 (Ottimale per architettura Hopper)

Le H200 sono basate sull'architettura NVIDIA Hopper. Anche se torch.float16 va benissimo per attivare i Tensor Cores, per testare al meglio 
l'architettura Hopper e' meglio utilizzare il formato BFloat16.
Puoi modificare questa riga nel tuo script:

    tensor_type = torch.bfloat16

Il consumo di memoria resterà identico (2 byte per elemento), ma sarà sfruttato l'utilizzo dell'hardware più moderno della GPU.

-------------
3) Abilitare TF32 (Opzionale)

Se vuoi testare i Tensor Cores simulando carichi di lavoro in singola precisione (FP32), l'architettura Hopper supporta il formato TF32 (TensorFloat-32).
Per farlo, dovresti impostare:

    tensor_type = torch.float32 

e aggiungere questa riga all'inizio dello script:

    torch.backends.cuda.matmul.allow_tf32 = True

Nota: questo raddoppierà l'uso della VRAM, quindi in quel caso dovrai dimezzare il BATCH_SIZE.

************************************************************************************

Come monitorare il test:
-------------
Mentre lo script è in esecuzione, apri un secondo terminale e lancia questo comando per monitorare il comportamento in tempo reale (si aggiorna ogni secondo):

    watch -n 1 nvidia-smi

Cosa controllare durante lo stress test:

Pwr:Usage/Cap: Dovrebbe passare da 65-70W attuali a circa  550W-600W. Se non raggiunge almeno i 500W, i Tensor Cores non sono saturati 
(ma con matrici da 32k x 32k dovresti raggiungerli facilmente).

Temp: Le temperature delle GPU durante il test saliranno rapidamente. Assicurarsi che si mantengano sotto gli 85C.  >) >) >)

GPU-Util: Dovrebbe arrivare costantentemente intorno al 100%.
