def main(model, test_obs, test_act):
    # read in test data
    # forward pass with model
    # display metrics, save results
    print("---------TESTING MODEL---------")
    model.evaluate(test_obs, test_act)

if __name__ == "__main__":
    main(model, test_obs, test_act)
