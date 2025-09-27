from lib import evaluation as ev

model = "LVAE"

runs = [1]
metric_lst = ['active_units_h02_z00', 'active_units_h02_z01', 'active_units_h02_z02',
              'active_units_h02_z03']#, 'active_units_h02_z04', 'active_units_h02_z05']
metric_lst = ['ami_h00', 'ami_h01', 'ami_h02', 'ami_h03', 'ami_h04']

ev.plot_hier_kl_loss(runs, model, num_plots=5)
ev.plot_loss(runs, model)
ev.plot_metrics(runs, model, metric_lst)
