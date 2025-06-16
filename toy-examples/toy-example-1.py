#%%
import elicito as el
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

#%%
class GenerativeModel:
    def __call__(self, prior_samples, design_matrix, n_gr):
        # extract prior samples per parameter type
        betas=prior_samples[:,:,:-1]
        sigma=prior_samples[:,:,-1][:,:,None]
        # linear predictor
        mu = tf.matmul(betas, design_matrix, transpose_b=True)
        # data-generating model
        y = tfd.Normal(loc=mu, scale=sigma).sample()
        # selected observations per group
        (y_gr0, y_gr1, y_gr2) = (y[:, :, i:i+30] for i in [0,30,60])
        return dict(y_gr0=y_gr0, y_gr1=y_gr1,
                    y_gr2=y_gr2, mu=mu, y=y)

# design-matrix for three-level dummy-coded predictor
def design_categorical(n_gr):
    incpt = [1]*3
    contrast_01 = [0, 1, 0]
    contrast_02 = [0, 0, 1]
    # contrast matrix
    c = tf.stack([incpt, contrast_01, contrast_02], axis=-1)
    # design matrix
    x = tf.concat([tf.broadcast_to(c[i, :], (n_gr, c.shape[1])) for
                   i in range(c.shape[0])], axis=0)
    return tf.cast(x, tf.float32)

model=el.model(
    obj=GenerativeModel,
    n_gr=30,
    design_matrix=design_categorical(n_gr=30)
)

parameters=[
    el.parameter(
        name=f"beta{i}",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper(f"mu{i}"),
        scale=el.hyper(f"sigma{i}", lower=0))
        ) for i in range(3)
    ]+[
    el.parameter(
        name="sigma",
        family=tfd.HalfNormal,
        hyperparams=dict(scale=el.hyper("sigma3", lower=0))
    ),
]

def r2(y, mu):
    return tf.divide(
        tf.math.reduce_variance(mu, axis=-1),
        tf.math.reduce_variance(y, axis=-1)
    )

targets=[
    el.target(
        name=f"y_gr{i}",
        query=el.queries.quantiles(quantiles=(.25, .50, .75)),
        loss=el.losses.MMD2(kernel="energy"),
        ) for i in range(3)
    ]+[
    el.target(
        name="r2",
        query=el.queries.quantiles(quantiles=(.25, .50, .75)),
        loss=el.losses.MMD2(kernel="energy"),
        target_method=r2,
        weight=10.
    )
]

expert = el.expert.data(
    dat={'quantiles_y_gr0': [0.64, 1.22, 1.89],
        'quantiles_y_gr1': [0.72, 1.39, 2.07],
        'quantiles_y_gr2': [0.76, 1.48, 2.22],
        'quantiles_r2': [0.07, 0.23, 0.61]}
    )

optimizer=el.optimizer(
    optimizer=tf.keras.optimizers.Adam,
    learning_rate=0.05
)

trainer=el.trainer(
    method="parametric_prior",
    seed=1234,
    epochs=600,
    progress=0
)

initializer=el.initializer(
    method="sobol",
    iterations=32,
    distribution=el.initialization.uniform(radius=2., mean=0.)
)

eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=expert,
    optimizer=optimizer,
    trainer=trainer,
    initializer=initializer
)
#%%
eliobj.fit(parallel=el.utils.parallel(runs=5))
#%%
eliobj.save(file="saved_eliobj/eliobj_param")
#eliobj = el.utils.load(file="saved_eliobj/eliobj_param.pkl")
#%%
el.plots.loss(eliobj, figsize=(6,2), save_fig="saved_plots/loss_param.png")
#%%
el.plots.hyperparameter(eliobj, figsize=(6,4), save_fig="saved_plots/hyperparam_param.png")
#%%
el.plots.elicits(eliobj, figsize=(6,2), save_fig="saved_plots/elicits_param.png")
#%%
el.plots.prior_marginals(eliobj, figsize=(6,2), save_fig="saved_plots/prior_marginals_param.png")
#%%
el.plots.prior_averaging(eliobj, figsize=(6,3.5), xlim_weights=0.3, save_fig="saved_plots/prior_averaging_param.png")