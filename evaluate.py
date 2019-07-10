def main(model, test_obs, test_act, test_agent):
    # read in test data
    # forward pass with model
    # display metrics, save results
    print("---------TESTING MODEL---------")
    print("Testing on " + test_agent)
    model.evaluate(test_obs, test_act)

if __name__ == "__main__":
    main(model, test_obs, test_act, test_agent)
