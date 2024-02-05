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
gpu_num=4
# for training general time predictor
for filename in filename_list:
    for seed in range(5):
        if 'Dataset1' in filename:
            num_neurons=6049
        elif 'Dataset2' in filename:
            num_neurons=7782
        elif 'Dataset3' in filename:
            num_neurons=7828
        elif 'Dataset4' in filename:
            num_neurons=6445
        elif 'Dataset5' in filename:
            num_neurons=8229
        elif 'Dataset6' in filename:
            num_neurons=8968

        cmd = f'CUDA_VISIBLE_DEVICES={cuda_id%gpu_num} python training_base.py {filename} {seed} {num_neurons}'
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
