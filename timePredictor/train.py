import subprocess

filename_list = [
        './SCNData/all_neuron_20210916.mat',        
        './SCNData/all_neuron_20210918.mat',
        './SCNData/all_neuron_20210922.mat']
cuda_id = 0
job_list = list()
cuda_id_list = list()

for seed in range(5):
    for filename in filename_list:
        if '0916' in filename:
            final_average_num=1816
        elif '0918' in filename:
            final_average_num=2335
        elif '0922' in filename:
            final_average_num=2350
        average_list = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 1000, 1500]
        average_list.extend([final_average_num-1, final_average_num])
        for average_num in list(reversed(average_list)):
            cmd = f'CUDA_VISIBLE_DEVICES={cuda_id%2} python CNN_average_neuron.py {filename} {seed} {average_num}'
            print(cmd)
            cuda_id += 1
            job = subprocess.Popen(cmd, shell=True)
            job_list.append(job)
            cuda_id_list.append(cuda_id)
            while len(job_list) >= 16:
                for i, job in enumerate(job_list):
                    try:
                        job.wait(16)
                        cuda_id = cuda_id_list[i]
                        cuda_id_list = cuda_id_list[:i] + cuda_id_list[i+1:]
                        job_list = job_list[:i] + job_list[i+1:]
                        break
                    except subprocess.TimeoutExpired as e:
                        continue
