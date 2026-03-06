Mission 1: Window Traversal

The Goal: The drone must autonomously navigate through a tunnel-like structure made of five square gates lined up in a row.

Our Solution: We tackle this using a straightforward, sequential vision-based pipeline called "Detect-Align-Pass".
Instead of trying to calculate a complex flight path for all five gates at once, the drone handles them one by one:

  Detect: The drone's camera acts as its eyes, searching for the very next gate in front of it.

  Align: Once spotted, the drone adjusts its position to center itself perfectly with the opening.

  Pass: It flies straight forward through the gate.

The drone simply repeats this three-step loop until it successfully clears the entire tunnel.


Mission 4: Dynamic Moving Landing

The Goal: The drone must track and safely land on a landing pad that is actively moving around.

Our Solution:
Landing on a moving target is tricky, so we use an "Adaptive Step-Descent" (or "Pounce") strategy. Here is how the drone's brain handles the chase and the landing:

 Smart Tracking: We use AI vision (YOLO) to spot the pad. Normally, the AI needs to be 65% sure it sees the pad to lock on. However, as the drone gets closer and the angle changes, we automatically drop that requirement to 50% so the drone doesn't accidentally lose sight of the target.

 Predictive Steering: Instead of just chasing where the pad is, the drone uses a steering algorithm (a Proportional controller with predictive gain) to aim for where the pad is going.

 Smooth Speed Reading: To figure out how fast the pad is moving across the camera's view, we use a math trick called an Exponential Moving Average. This simply smooths out the speed calculations so the drone's movements stay stable and don't get jerky.

 The "Kill-Zone" Landing: The drone hovers and tracks the pad until it is perfectly aligned within an 80-pixel center area on the camera view. Once the pad enters this "Kill-Zone", the drone immediately triggers the descent to "pounce" and land!
