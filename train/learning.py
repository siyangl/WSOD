import logging
import time
from datetime import datetime
import os
import numpy as np
import inspect
import tensorflow as tf


def train(graph, logdir,
          loss, optimizer,
          variables_to_train, global_step,
          num_steps, log_interval=20, summary_interval=100, snapshot_interval=10000,
          sess_config=None,
          variables_to_restore=None, restore_ckpt=None,
          saver=None):
  # log setting:
  log_filename = os.path.join(logdir, 'train_%s.log' % datetime.now())
  if not os.path.exists(logdir):
    os.mkdir(logdir)
  logging.basicConfig(filename=log_filename, level=logging.DEBUG)
  frame = inspect.currentframe()
  args, _, _, values = inspect.getargvalues(frame)
  logging.info('Train with args:')
  for arg in args:
    if isinstance(values[arg], list):
      logging.info("    %s:"%arg)
      for v in values[arg]:
        logging.info("    %s"%v.name)
    else:
      logging.info("    %s = %s" % (arg, values[arg]))
    # print

  # training op
  with tf.device('/gpu:0'):
    grads = optimizer.compute_gradients(loss, variables_to_train)
    grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  # grad summary
  for grad in grads:
    tf.histogram_summary(grad[0].name, grad[0])
  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(logdir, graph=graph)

  if not sess_config:
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
  sess = tf.Session(graph=graph, config=sess_config)

  # Init all variables
  sess.run(tf.initialize_all_variables())
  sess.run(tf.initialize_local_variables())
  if variables_to_restore and restore_ckpt:
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, restore_ckpt)
  if not saver:
    saver = tf.train.Saver(tf.trainable_variables())  # need to update for model with batchnorm

  # train

  tf.train.start_queue_runners(sess=sess)
  step = 0
  while step <= num_steps:
    start_time = time.time()
    _, loss_value, global_step_out = sess.run([grad_updates, loss, global_step])
    duration = time.time() - start_time
    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % log_interval == 0:
      # examples_per_sec = FLAGS.batch_size / float(duration)
      format_str = ('%s: step %d, loss = %.2f (%.3f sec/batch)')
      print(format_str % (datetime.now(), step, loss_value, duration))
      logging.info(format_str % (datetime.now(), step, loss_value, duration))

    if step % summary_interval == 0:
      summary_str = sess.run(summary_op)
      summary_writer.add_summary(summary_str, step)

    # Save the model checkpoint periodically.
    if (step % snapshot_interval == 0 or step >= num_steps) and step > 0:
      checkpoint_path = os.path.join(logdir, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=step)
    step += 1