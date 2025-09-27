import lib.evaluation as ev


model = "iVAE"
runs = [4]

metric_lst = ['approx_mi_xz','approx_mi_xz','approx_mi_xz']

# ev.plot_loss(runs, model)
ev.plot_metrics(runs, model, metric_lst)
