import subprocess

cuda_id = 0
job_list = list()
cuda_id_list = list()
gpu_num=4

for seed in range(5):
    for spatial_class in [1, 2, 3]:
        average_list = [1, 10, 30, 50, 100, 300, 351-1, 500, 600, 606-1,  700-1, 700,  750,  800,  850, 860-1, 900, 1000, 1167-1, 1210-1, 1719-1]
        
        for num_neuron in average_list:
            cmd = f'CUDA_VISIBLE_DEVICES={cuda_id%gpu_num} python training_base_1for3.py {spatial_class} {seed} {num_neuron}'
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
