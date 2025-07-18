# Code base for ICLR submission

We use composers and custom model from MoEUT for training.
We will use streaming datasets with pre-tokenised datasets.

We will AdamW and Muon as baselines

What will be the abbie baselies?

I guess 


Todo:
    1. Finish the huggingface integration so we can upload to the hub
    2. Test that the trainer is running
    3. Compare versus the old moeut code to make sure nothing has changed!
    3.1 Add the other rope
    3.2 make sure that it is fine!
    4. Implement our unique features
    4.1 check that they work as designed
        Devise a method to check and do it!


## For the MoE
    MoEUT vs DeepSeek
    Attention routing vs not
    Expert needs to decide when to leave -- mixture of depths -- read and implements either second expert or one of the shared?


## The KV cache

