# Code base for ICLR submission

We use composers and custom model from MoEUT for training.
We will use streaming datasets with pre-tokenised datasets.

We will AdamW and Muon as baselines

What will be the abbie baselies?

I guess 

Todo:
    1. Finish the huggingface integration so we can upload to the hub
        Composer + huggingface is intergraged √
        Figure out how tokization works and tokenize the dataset -- input ids?
            -- We will use the StreamingTextDataset class to handle the data streaming -- it can tokenize but we will make sure to used pre-tokenized outputs
            -- We reserve the option to have embeddings as well for diffusion in the future.
            -- Sequence IDs seems to be too far-fetched at this point, but we retain the options
        Figure out the loss
    2. Test that the trainer is running √
        Trainer is indeed running well
    3. Compare versus the old moeut code to make sure nothing has changed! √
    4. Implement our unique features
        - Infinite looping
            infinite while loop.
            eachtime have a condition on whether to contine
            Flip condition to false to exit
                -- per token is posible by viewing the queries without the finished tokens -- this makes the attention roughly 50% faster.
                -- we then also need to scale the summing to be token based
                -- we need a partial mlp somehow? -- just nuke x! simple !!! remove the indeces from the Q and later X of the MLP and scatter back! -- figure out the scatter add op.
                
                Make a cute toy example!

            
               We need a special loss now that will do two things -- make the the experts used equally kinda and punish later layers!
               Generalist 


        - Peri norm the blocks with RMS -- 5 mins√ [Not done we need it per token] -- probably done
        - Cumulative average: value = [(count - 1 ) /  count] * value + x / count -- make the contrib of every block the same -- 5 mins -- done √ [Not done we need it per token]
        -- do the abbie loop and embedding -- Middle-Cycle Strategy
        -- Shall we add a residual? -- how to normalise
        -- What about the KV cache

    4.1 check that they work as designed
        Devise a method to check and do it!

Add Muon

## For the MoE
    MoEUT vs DeepSeek
    Attention routing vs not
    Expert needs to decide when to leave -- mixture of depths -- read and implements either second expert or one of the shared?


## The KV cache
    From the Mixture of Reccurence paper

