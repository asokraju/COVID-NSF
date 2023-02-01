# COVID-NSF: Custom PPO algorithm with multi-threading for data collection implemented in TensorFlow

The code repository contains the implementation of PPO clip algorithm, which is a popular reinforcement learning algorithm, written in TensorFlow. Reinforcement learning algorithms typically have three stages: data collection, finding gradients, and updating the policy network. The last two stages are typically done using deep learning packages such as TensorFlow and PyTorch and are implemented on a GPU. However, the first stage, data collection, is typically done on a CPU, which can slow down the overall process.

To overcome this limitation and improve the overall speed of the algorithm, I have used multi-threading through the Concurrent Python toolbox. Multi-threading allows the data collection process to be executed simultaneously with the other stages of the algorithm, making the overall process more efficient. 


This is  used for computing the optimal policies in SEIR epidamic models. During the pandemic different control policies such as Lockdown, social distancing, and Congregation restriction has been implemented in an ad-hoc manner to control the disease spread while minimizing the economic/social losses. However, the control policies should be designed to balance their effectiveness on disease spreading while minimizing their impacts on economy/society, poses the key challenge in pandemic-time decision making.  I study and propose the optimal control policies through Reinforcement Learning under data uncertainity and sampling bias. To achieve this I have used the Proximal Policy Optimization algorithm, implemented using TensorFlow deeplearning toolbox. 

(details:)

- We model the spread of COVID-19 as an SEIR (Susceptible, Exposed, Infected, and Removed) epidemic model.
- The transmission rate `beta`, representing the number of susceptible people infected per infectious person per day, is a controllable variable.
- The decision makers can choose from three policies - Lockdown, Social Distancing, or Open economy - which determine the value of `beta`.
- The decision makers aim to minimize the public health cost (proportional to the total number of infected people) and the economic cost associated with the chosen policy. For example, the economic cost is assumed to be 1 for a lockdown, 0.25 for social distancing, and 0 for an open economy.
- In the literature, various control techniques are provided to achieve this, but they use raw case data (the SEIR values) to determine the policy.
- This study aims to answer two questions:
  1. How does under-reporting of case data affect the cost, if the policy makers are interested in implementing the true optimal controller?
  2. How does correcting for under-reporting, based on guesses, affect the cost if the policy makers are aware of it?
- The true optimal solution is intractable, so we use Reinforcement Learning, specifically the Proximal Policy Optimization (PPO) algorithm, to compute it numerically.
- We perform a sensitivity analysis of the cost with respect to under-reported case data by adding noise to the states while predicting the control input from the policy.
- Our results show that the cost is highly sensitive when under-reporting is low, but becomes insensitive after a certain threshold of under-reporting.


`In summary, I have created a custom implementation of the PPO algorithm in TensorFlow that utilizes multi-threading for data collection. The use of multi-threading improves the efficiency of the algorithm and the TensorFlow library allows for easy implementation of the PPO algorithm's necessary components such as the actor-critic network, loss function, and optimizer. This implementation is intended for use in large-scale reinforcement learning tasks, specifically in the context of pandemic decision making and finding optimal control policies for disease spread.`
