import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.pde = ml_collections.ConfigDict()
    config.pde.n_grid = 70
    config.pde.nt = 1000
    config.pde.dx = 10.0
    config.pde.dt = 0.001
    config.pde.nbc = 120
    config.pde.f = 15.0
    config.pde.sz = 10
    config.pde.gz = 10
    config.pde.ng = 70
    config.pde.ns = 5

    config.model = ml_collections.ConfigDict()
    config.model.dim = 64
    config.model.dim_mults = (1, 2, 4, 8)
    config.model.flash_attn = False
    config.model.channels = 1

    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.image_size = 72
    config.diffusion.timesteps = 1000
    config.diffusion.sampling_timesteps = 250
    config.diffusion.objective = 'pred_noise'
    config.diffusion.model_path = 'pretrained_models/model-4.pt'

    config.optimization = ml_collections.ConfigDict()
    config.optimization.lr = 0.03
    config.optimization.ts = 300
    config.optimization.diffusion_ts = 1
    config.optimization.regularization = 'diffusion'
    config.optimization.reg_lambda = 0.75
    config.optimization.use_time_weight = False
    config.optimization.fixed_timestep = None  # Maximum timestep for diffusion regularization (None = use model's max, e.g., 500 limits to [0, 500))
    config.optimization.sigma = 10.0
    config.optimization.sigma_x0 = 0.0001
    config.optimization.initial_type = 'smoothed'
    config.optimization.noise_std = 0.0
    config.optimization.noise_type = 'gaussian'
    config.optimization.missing_number = 0


    config.data = ml_collections.ConfigDict()
    config.data.seismic_data_dir = 'dataset/OpenFWI/Seismic_Data/'
    config.data.velocity_data_dir = 'dataset/OpenFWI/Velocity_Data/'
    config.data.batch_size = 1
    config.data.data_pattern = '*.npy'
    config.data.use_mmap = True

    config.experiment = ml_collections.ConfigDict()
    config.experiment.name = 'red_diffeq_default'
    config.experiment.results_dir = 'experiment/'
    config.experiment.save_intermediate = False
    config.experiment.log_interval = 10
    config.experiment.save_metrics = True
    config.experiment.random_seed = None

    return config

def get_marmousi_config():
    config = get_config()
    config.data.seismic_data_dir = 'dataset/Marmousi/Seismic_Data/'
    config.data.velocity_data_dir = 'dataset/Marmousi/Velocity_Data/'
    config.data.batch_size = 1
    config.experiment.name = 'marmousi_inversion'
    return config
