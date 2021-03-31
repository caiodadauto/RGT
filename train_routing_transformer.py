import argparse

import pytop

import tensorflow as tf
from tensorflow.compat.v1.train import polynomial_decay
from sonnet.optimizers import Adam, Momentum

from routergn.supervised.training import train
from routergn.supervised.models import RoutingGraphTransformer


def optimizer(s):
    opts = dict(adam=Adam, sgd=Momentum)
    if s not in opts:
        raise argparse.ArgumentTypeError("Support only adam or sgd parameter")
    return opts[str.casefold(s)]


def weights(s):
    try:
        l = list(map(float, s.split(",")))
    except:
        raise argparse.ArgumentTypeError("The format has to be <w1>,<w2>")
    if len(l) != 2:
        raise argparse.ArgumentTypeError("Only two floats are allowed")
    return tf.constant(l)


def run(
    train_path,
    # validation_path,
    delta_log,
    seed,
    epochs,
    batch_size,
    msgs,
    heads,
    optimizer,
    init_lr,
    last_lr,
    steps_lr,
    polynomial_power,
    class_weights,
):
    batch_generator = pytop.generator.batch_files_generator(
        train_path, batch_size, bidim_solution=False
    )
    model = RoutingGraphTransformer(
        num_of_msg=msgs, num_of_heads_core=heads, num_of_heads_routing=heads
    )
    global_step = tf.Variable(0, trainable=False)
    learning_rate = polynomial_decay(
        init_lr,
        global_step,
        decay_steps=steps_lr,
        end_learning_rate=last_lr,
        power=polynomial_power,
    )
    opt = optimizer(learning_rate)
    # train(model, opt, batch_generator, epochs, class_weights, delta_log)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "train_path",
        type=str,
        help="Path to the train dataset",
    )
    # p.add_argument(
    #    "validation_path",
    #    type=str,
    #    help="Path to the validation dataset",
    # )
    p.add_argument(
        "--delta-log",
        type=float,
        default=30,
        help="Elapsed time to log",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed used for both random states, tensorflow and numpy",
    )
    p.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size to be used in the training",
    )
    p.add_argument(
        "--msgs",
        type=int,
        default=20,
        help="The number of mesages used for massage passing",
    )
    p.add_argument(
        "--heads",
        type=int,
        default=5,
        help="The number of heads used in the multihead attention",
    )
    p.add_argument(
        "--optimizer",
        type=optimizer,
        default=Adam,
        help="The optimizer that will be used",
    )
    p.add_argument(
        "--init-lr",
        type=float,
        default=5e-3,
        help="Initial learning rate for polonomial decay",
    )
    p.add_argument(
        "--last-lr",
        type=float,
        default=5e-5,
        help="Last learning rate for polonomial decay",
    )
    p.add_argument(
        "--steps-lr",
        type=float,
        default=500000,
        help="Number of steps to decrease the initial to the final learning rate",
    )
    p.add_argument(
        "--polynomial-power",
        type=float,
        default=3,
        help="The degree of the polynomial function"
        " responsible to decrease the learning rate",
    )
    p.add_argument(
        "--class-weights",
        type=weights,
        default=tf.constant([1.0, 1.0]),
        help="The weights for each class applied to the loss function",
    )
    args = p.parse_args()
    run(**vars(args))
