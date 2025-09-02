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



# Where we are today:
    1. Model is running and training - but not in it's final form
    2. Dynamic halting needs to be implemented
    3. Per token rescaling
    4. FSDP and firends
    5. KV cache from MoR


Add Muon

## For the MoE
    MoEUT vs DeepSeek
    Attention routing vs not
    Expert needs to decide when to leave -- mixture of depths -- read and implements either second expert or one of the shared?


## The KV cache
    From the Mixture of Reccurence paper



The loss function should balance itself. For better science I need some hypothesis to see if this will work.
1 . Tau is multiplied together with S into each token. works really well I think.
2. Tau is both multiplied at each token and loss. Not great
3. Tau is just multipled with the loss



New masking for the attention
    Take the regular mask and just assembled it as before


    
    

pos_offset and kv caching 


1 expert case is handled

Add all hyper params like learned tau and bias to the hf checkpoint

How do we verify everything

    Plot entropy and maybe seq length as a graph that can scroll and as a total graph -- done

    Plot expert selection -- almost done
        -- see that it works with microbatching

    Do the same reg loss as they had
        -- will do

    Make ACT for comparisons 
        -- will do
    

    Paramterise everything and run

    Fix initialisation
    batch packing

    Shanon entropy being 10 seems highly unlikely
        -- make shanon entropy for final token as well
        -- make shanon entropy work as the expert select
        -- make exit entropy the same and show final entropy as well

    
    Exit expert -- might be very good

        -- we maybe we can look at the entropy

    For a policy
        -- give it more info 
    use RL to improve the router's descicion process
    Temporal iterations
    Multiple rollouts
        -- route several times the same tokens


    Visualise the attention mask

Entropy calc before applying sigmoid apparently1!!


Mixture of Reccurance routing examples




Parameters:
    Token exit
    Peri-LN
    Attn experts
    FFN experts
    Shared experts
    Expert balancing
    Early exiting losses

Future work:
    Multi Token prediction
    The cool attention from Meta
    Dynamic number of experts per iteration


Where will I handle the rescaling
    In the main body I guess
    What about the norming, I think it should be hidden in the modules
    


TODO:

Take a break
    Check the end of trainig time and make sure to start documenting before end of today
    5. Document everything
    1. Make better naming scheme
        Make sure entropy works with Transformer model?
    2. Look into the early exit
    3. Implement Geiping
    4. Fix the other balancing methods

What can be the issue
    Attention mask --good
    Position mask -- good
    RoPE
    Norms and application of blocks --good
    Metrics some how affect gradient -- mhmmm

    Both transformer with simple and moeut with moeut layers gets messed up so likley it is the norming and block application



srun -w ruapehu -c 24 --gres=gpu:0 --partition=interactive --pty bash

find . -user "$USER" -type f -delete 2>/dev/null