import torch

torch.manual_seed(0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: {}".format(device))

    # Model Parameters

    print("Program has Ended")
