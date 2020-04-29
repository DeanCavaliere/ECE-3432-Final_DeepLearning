import torch

testCuda = torch.cuda.is_available()

if(testCuda):
    NumCuda = torch.cuda.device_count()
    for i in range(NumCuda):
        dummy = torch.cuda.get_device_properties(i)
        print('\n')
        print('Cuda ID:                 ' + 'cuda:' + str(i))
        print('Name:                    ' + str(dummy.name))
        print('Total Memory:            ' + str(round(dummy.total_memory/1e9,1))+ ' GB')
        print('Multiprocessor Count:    ' + str(dummy.multi_processor_count))

else:
    print('No Cuda Device Found')