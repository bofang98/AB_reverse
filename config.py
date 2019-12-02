
params = dict()

params['data'] = 'UCF-101'
params['dataset'] = '/data2/data/video_data/UCF-101'

params['num_classes'] = 15

params['epoch_num'] = 300
params['batch_size'] = 4
params['num_workers'] = 4
params['learning_rate'] = 0.001
params['step'] = 10
params['momentum'] = 0.9
params['weight_decay'] = 0.0005
params['display'] = 10

params['pretrained'] = None
params['gpu'] = [0]
params['log'] = 'log'
params['save_path_base'] = '/home/fb/project/AB_reverse/'
