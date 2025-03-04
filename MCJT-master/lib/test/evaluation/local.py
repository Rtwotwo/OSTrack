from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    # settings.got10k_lmdb_path = '/media/estar/Data/LY/Anti-UAV/img/got10k_lmdb'
    # settings.got10k_path = '/media/estar/Data/LY/Anti-UAV/img/got10k'
    # settings.got10k_path = '/media/estar/Data/LY/1st-test_dev/img/got-10k'
    settings.got10k_path = '/media/estar/Data/LY/2nd-test_dev/img/got-10k'
    # settings.got_packed_results_path = ''
    # settings.got_reports_path = ''
    # settings.itb_path = '/media/estar/Data/LY/Anti-UAV/img/itb'
    # settings.lasot_extension_subset_path_path = '/media/estar/Data/LY/Anti-UAV/img/lasot_extension_subset'
    # settings.lasot_lmdb_path = '/media/estar/Data/LY/Anti-UAV/img/lasot_lmdb'
    # settings.lasot_path = '/media/estar/Data/LY/Anti-UAV/img/lasot'
    settings.network_path = '/media/estar/Data/LY/Models/Anti-UAV/Ostrack-New/tracking/output/test/networks'    # Where tracking networks are stored.
    # settings.nfs_path = '/media/estar/Data/LY/Anti-UAV/img/nfs'
    # settings.otb_path = '/media/estar/Data/LY/Anti-UAV/img/otb'
    settings.prj_dir = '/media/estar/Data/LY/Models/Anti-UAV/Ostrack-New/tracking'
    settings.result_plot_path = '/media/estar/Data/LY/Models/Anti-UAV/Ostrack-New/tracking/output/test/result_plots'
    settings.results_path = '/media/estar/Data/LY/Models/Anti-UAV/Ostrack-New/tracking/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/media/estar/Data/LY/Models/Anti-UAV/Ostrack-New/tracking/output'
    settings.segmentation_path = '/media/estar/Data/LY/Models/Anti-UAV/Ostrack-New/tracking/output/test/segmentation_results'
    # settings.tc128_path = '/media/estar/Data/LY/Anti-UAV/img/TC128'
    # settings.tn_packed_results_path = ''
    # settings.tnl2k_path = '/media/estar/Data/LY/Anti-UAV/img/tnl2k'
    # settings.tpl_path = ''
    # settings.trackingnet_path = '/media/estar/Data/LY/Anti-UAV/img/trackingnet'
    # settings.uav_path = '/media/estar/Data/LY/Anti-UAV/img/uav'
    # settings.vot18_path = '/media/estar/Data/LY/Anti-UAV/img/vot2018'
    # settings.vot22_path = '/media/estar/Data/LY/Anti-UAV/img/vot2022'
    # settings.vot_path = '/media/estar/Data/LY/Anti-UAV/img/VOT2019'
    settings.youtubevos_dir = ''

    return settings

