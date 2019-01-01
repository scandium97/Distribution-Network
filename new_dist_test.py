from new_train import export_results
from model import cdf_reg_model
from api.alpha_data_api import AlphaDataApi, generate_handles_by_dates

if __name__ == '__main__':
    train_handle, valid_handle, test_handle = generate_handles_by_dates(
        ["20141231", "20151231", "20161231"], n_class, 42, 5)
    model = cdf_reg_model(isTraining=False)
    export_results(model,test_handle,name="t")

