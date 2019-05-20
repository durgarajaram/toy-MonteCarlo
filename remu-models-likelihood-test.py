from six import print_
import numpy as np
from matplotlib import pyplot as plt
from remu import binning
from remu import likelihood

response_matrix = np.load("input/response_matrix.npy")
generator_truth = np.load("input/truth.npy")

with open("config/true_bins.yml", 'rt') as f:
    true_bins = binning.yaml.load(f)

# real data
with open("config/recon_bins.yml", 'rt') as f:
    recon_bins = binning.yaml.load(f)
    
recon_bins.fill_from_csv_file("input/real_data.txt")
data = recon_bins.get_entries_as_ndarray()

lm = likelihood.LikelihoodMachine(data, response_matrix, truth_limits=generator_truth, limit_method='prohibit')

true_bins.fill_from_csv_file("input/model-1-truth.txt")
model1 = true_bins.get_values_as_ndarray()
model1 /= np.sum(model1)
true_bins.reset()
true_bins.fill_from_csv_file("input/model-2-truth.txt")
model2 = true_bins.get_values_as_ndarray()
model2 /= np.sum(model2)

with open("output/simple_hypotheses.txt", 'w') as f:
    print_(lm.log_likelihood(model1*1000), file=f)
    print_(lm.likelihood_p_value(model1*1000), file=f)
    print_(lm.log_likelihood(model2*1000), file=f)
    print_(lm.likelihood_p_value(model2*1000), file=f)

with open("output/model1_fit.txt", 'w') as f:
    model1_shape = likelihood.TemplateHypothesis([model1])
    retA = lm.max_log_likelihood(model1_shape)
    print_(retA, file=f)

with open("output/model2_fit.txt", 'w') as f:
    model2_shape = likelihood.TemplateHypothesis([model2])
    retB = lm.max_log_likelihood(model2_shape)
    print_(retB, file=f)

with open("output/fit_p-values.txt", 'w') as f:
    print_(lm.max_likelihood_p_value(model1_shape, nproc=4), file=f)
    print_(lm.max_likelihood_p_value(model2_shape, nproc=4), file=f)

figax = recon_bins.plot_values(None, kwargs1d={'label': 'data', 'color': 'k'})
model1_reco = response_matrix.dot(model1_shape.translate(retA.x))
model2_reco = response_matrix.dot(model2_shape.translate(retB.x))
recon_bins.plot_ndarray(None, model1_reco, kwargs1d={'label': 'model 1', 'color': 'b'}, sqrt_errors=True, error_xoffset=-0.1, figax=figax)
recon_bins.plot_ndarray("reco-comparison.png", model2_reco, kwargs1d={'label': 'model 2', 'color': 'r'}, sqrt_errors=True, error_xoffset=+0.1, figax=figax)

with open("output/mix_model_fit.txt", 'w') as f:
    mix_model = likelihood.TemplateHypothesis([model1, model2])
    ret = lm.max_log_likelihood(mix_model)
    print_(ret, file=f)

with open("output/mix_model_p_value.txt", 'w') as f:
    print_(lm.max_likelihood_p_value(mix_model, nproc=4), file=f)

p_values = []
A_values = np.linspace(0, 1000, 11)
for A in A_values:
    fixed_model = mix_model.fix_parameters((A, None))
    p = lm.max_likelihood_ratio_p_value(fixed_model, mix_model, nproc=4)
    print_(A, p)
    p_values.append(p)

wilks_p_values = []
fine_A_values = np.linspace(0, 1000, 100)
for A in fine_A_values:
    fixed_model = mix_model.fix_parameters((A, None))
    p = lm.wilks_max_likelihood_ratio_p_value(fixed_model, mix_model)
    print_(A, p)
    wilks_p_values.append(p)

fig, ax = plt.subplots()
ax.set_xlabel("Model 1 weight")
ax.set_ylabel("p-value")
ax.plot(A_values, p_values, label="Profile plug-in")
ax.plot(fine_A_values, wilks_p_values, label="Wilks")
ax.axvline(ret.x[0], color='k', linestyle='solid')
ax.axhline(0.32, color='k', linestyle='dashed')
ax.axhline(0.05, color='k', linestyle='dashed')
ax.legend(loc='best')
fig.savefig("p-values.png")
