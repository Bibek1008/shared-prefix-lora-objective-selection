# Compute-Efficient LoRA Objective Selection with Shared Prefixes

A controlled experiment investigating whether a shared training prefix can
reduce the cost of comparing token-mean and example-mean LoRA objectives.

Normally, comparing two LoRA objectives requires training two complete models:

- Token-mean: 576 updates
- Example-mean: 576 updates
- Total: 1,152 updates

This study instead trains a shared 288-update prefix, preserves its exact
training state, and continues two 288-update branches:

```text
                 Shared prefix
                  288 updates
                       |
              ┌────────┴────────┐
              ↓                 ↓
       Token-mean branch  Example-mean branch
          288 updates         288 updates