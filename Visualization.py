import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfc
from tensorboard.plugins import projector

class TF_visualizer(object):
    def __init__(self, dimension, vecs_file, metadata_file, output_path):
        self.dimension = dimension
        self.vecs_file = vecs_file
        self.metadata_file = metadata_file
        self.output_path = output_path
        
        self.vecs = []
        with open(self.vecs_file, 'r') as vecs:
            for i, line in enumerate(vecs):
                if line != '': self.vecs.append(line)

    def visualize(self):
        # adding into projector
        config = projector.ProjectorConfig()

        placeholder = np.zeros((len(self.vecs), self.dimension))
        
        for i, line in enumerate( self.vecs ):   
            placeholder[i] = np.fromstring(line, sep=',')

        embedding_var = tfc.Variable(placeholder, trainable=False, name='metadata')

        embed = config.embeddings.add()
        embed.tensor_name = embedding_var.name
        embed.metadata_path = self.metadata_file

        # define the model without training
        sess = tfc.InteractiveSession()
        
        tfc.global_variables_initializer().run()
        saver = tfc.train.Saver()
        
        saver.save(sess, os.path.join(self.output_path, 'w2x_metadata.ckpt'))

        writer = tfc.summary.FileWriter(self.output_path, sess.graph)
        projector.visualize_embeddings(writer, config)
        sess.close()
        print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(self.output_path))


