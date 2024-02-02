import subprocess

filename_list = [
        '../SCNData/Dataset1_SCNProject.mat',
        '../SCNData/Dataset2_SCNProject.mat',
        '../SCNData/Dataset3_SCNProject.mat',
        '../SCNData/Dataset4_SCNProject.mat',
        '../SCNData/Dataset5_SCNProject.mat',
        '../SCNData/Dataset6_SCNProject.mat',
        ]
        
cuda_id = 0
job_list = list()
cuda_id_list = list()
gpu_num=8
# for training general time predictor
for filename in filename_list:
    for seed in range(5):
        
        num_neuron_list = [1, 10, 30, 50, 100, 300, 500, 600,  700,  750,  800,  850,  900,  950, 1000, 1500]
        for num_neuron in num_neuron_list:
            cmd = f'CUDA_VISIBLE_DEVICES={cuda_id%gpu_num} python training_base.py {filename} {seed} {num_neuron}'
            print(cmd)
            cuda_id += 1
            job = subprocess.Popen(cmd, shell=True)
            job_list.append(job)
            cuda_id_list.append(cuda_id)
            while len(job_list) >= gpu_num:
                for i, job in enumerate(job_list):
                    try:
                        job.wait(gpu_num)
                        cuda_id = cuda_id_list[i]
                        cuda_id_list = cuda_id_list[:i] + cuda_id_list[i+1:]
                        job_list = job_list[:i] + job_list[i+1:]
                        break
                    except subprocess.TimeoutExpired as e:
                        continue

# for the general time predictor test on the testing tests with 5 submodules.
# we take 'Dataset1_SCNProject.mat' as example.               
for seed in range(5):
    num_neuron_list = [143-1, 238-1, 400-1, 448-1, 586-1]
    for num_neuron in num_neuron_list:
        cmd = f'CUDA_VISIBLE_DEVICES={cuda_id%gpu_num} python training_base.py "../SCNData/Dataset1_SCNProject.mat" {seed} {num_neuron}'
        print(cmd)
        cuda_id += 1
        job = subprocess.Popen(cmd, shell=True)
        job_list.append(job)
        cuda_id_list.append(cuda_id)
        while len(job_list) >= gpu_num:
            for i, job in enumerate(job_list):
                try:
                    job.wait(gpu_num)
                    cuda_id = cuda_id_list[i]
                    cuda_id_list = cuda_id_list[:i] + cuda_id_list[i+1:]
                    job_list = job_list[:i] + job_list[i+1:]
                    break
                except subprocess.TimeoutExpired as e:
                    continue