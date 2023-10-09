import tensorflow as tf

from sensory import SensorySystem
from compute import ComputeSystem

device = "cuda" if tf.test.is_gpu_available() else "cpu"

# Define the input placeholders for sensory inputs
visual_input = tf.placeholder(tf.float32, shape=[None, visual_input_shape], name='visual_input')
auditory_input = tf.placeholder(tf.float32, shape=[None, auditory_input_shape], name='auditory_input')
somatosensory_input = tf.placeholder(tf.float32, shape=[None, somatosensory_input_shape], name='somatosensory_input')

# This is the model training learning rate
TRAINING_LEARNING_RATE = 0.001

# There should be another learning rate for the human which is a function of the human's age,
# other factors, possibly modeled as a function & hence another small network? Inputs?

# Define the Attention module. TODO: Fix the compute_output circular dependency
def attention_module(sensory_output, compute_output = None):
    # Implement attention mechanism here
    attention_weights = tf.nn.softmax(tf.matmul(sensory_output, compute_output), axis=1)
    return attention_weights

# Define the Subconscious module
def subconscious_module(attention_weights, sensory_output, compute_output):
    # Implement the subconscious processing based on attention and sensory data
    # Output subconscious reactions and moral judgments
    model = tf.layers.dense([attention_weights, sensory_output, compute_output], 32, activation=None)
    return model

# Define the Reward Function module
def reward_function(compute_output, subconscious_output):
    # Implement the novel reward function as described in the model
    # Calculate rewards based on actions and subconscious judgments
    model = tf.layers.dense([compute_output, subconscious_output], 32, activation=None)
    return model

# Define the Motor Output module
def motor_output(actions, subconscious_reactions):
    # Implement the motor output module to translate actions into external reactions
    model = tf.layers.dense([actions, subconscious_reactions], 32, activation=None)
    return model

# Build the computational graph
sensory_module = SensorySystem()
sensory_output = sensory_module.process(visual_input, auditory_input, somatosensory_input)

attention_weights = attention_module(sensory_output)

compute_module = ComputeSystem()
compute_output = compute_module.process(sensory_output, attention_weights)

subconscious_output = subconscious_module(attention_weights, sensory_output, compute_output)
rewards = reward_function(compute_output, subconscious_output)
actions = motor_output(compute_output, subconscious_output)

# Define loss and optimization (specifics would depend on training objectives)
loss = tf.reduce_sum(rewards)
optimizer = tf.train.AdamOptimizer(learning_rate=TRAINING_LEARNING_RATE).minimize(loss)

