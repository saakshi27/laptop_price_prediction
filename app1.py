import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
def main():
    html_temp = """
        <div style ="background-color:lightblue;padding:16px">
        <h2 style ="color:black;text-align:center;">
        Car Price Prediction 
        </div>
    """

    model = xgb.XGBRegressor()
    model.load_model('xgb_model.json')
    st.markdown(html_temp, unsafe_allow_html=True)

    companies = ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI', 'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer', 'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG']
    types = ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook']
    cpu_brands = ['Intel Core i5', 'Intel Core i7', 'AMD Processors', 'Intel Core i3', 'Other Intel Processors']
    gpu_brands = ['Intel', 'AMD', 'Nvidia']
    os_options = ['Mac', 'Others/No OS/Linux', 'Windows']

    company = st.selectbox('Brand', companies)
    laptop_type = st.selectbox('Type', types)
    ram = st.selectbox('RAM (in GB)', [2, 4, 8, 16, 32, 64])
    weight = st.number_input('Weight of the Laptop')
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS', ['No', 'Yes'])
    screen_size = st.number_input('Screen Size')
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    cpu_brand = st.selectbox('CPU Brand', cpu_brands)
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    gpu_brand = st.selectbox('GPU Brand', gpu_brands)
    os_option = st.selectbox('OS', os_options)

    # Preprocess input data
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = round(((X_res ** 2) + (Y_res ** 2)) ** 0.5 / (screen_size+0.0000001), 6)
    print(ppi)
    import pandas as pd
    data_new = pd.DataFrame({
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'IPS': ips,
        'ppi': ppi,
        'HDD': hdd,
        'SSD': ssd,
        'Company_Acer': [1 if company == 'Acer' else 0],
        'Company_Apple': [1 if company == 'Apple' else 0],
        'Company_Asus': [1 if company == 'Asus' else 0],
        'Company_Chuwi': [1 if company == 'Chuwi' else 0],
        'Company_Dell': [1 if company == 'Dell' else 0],
        'Company_Fujitsu': [1 if company == 'Fujitsu' else 0],
        'Company_Google': [1 if company == 'Google' else 0],
        'Company_HP': [1 if company == 'HP' else 0],
        'Company_Huawei': [1 if company == 'Huawei' else 0],
        'Company_LG': [1 if company == 'LG' else 0],
        'Company_Lenovo': [1 if company == 'Lenovo' else 0],
        'Company_MSI': [1 if company == 'MSI' else 0],
        'Company_Mediacom': [1 if company == 'Mediacom' else 0],
        'Company_Microsoft': [1 if company == 'Microsoft' else 0],
        'Company_Razer': [1 if company == 'Razer' else 0],
        'Company_Samsung': [1 if company == 'Samsung' else 0],
        'Company_Toshiba': [1 if company == 'Toshiba' else 0],
        'Company_Vero': [1 if company == 'Vero' else 0],
        'Company_Xiaomi': [1 if company == 'Xiaomi' else 0],
        'TypeName_2 in 1 Convertible': [1 if company == '2 in 1 Convertible' else 0],
        'TypeName_Gaming':[1 if company == 'Gaming' else 0],
        'TypeName_Netbook': [1 if company == 'Netbook' else 0],
        'TypeName_Notebook': [1 if company == 'Notebook' else 0],
        'TypeName_Ultrabook': [1 if company == 'Ultrabook' else 0],
        'TypeName_Workstation': [1 if company == 'Workstation' else 0],
        'Cpu_Brand_AMD Processors': [1 if company == 'AMD Processors' else 0],
        'Cpu_Brand_Intel Core i3': [1 if company == 'Intel Core i3' else 0],
        'Cpu_Brand_Intel Core i5': [1 if company == 'Intel Core i5' else 0],
        'Cpu_Brand_Intel Core i7': [1 if company == 'Intel Core i7' else 0],
        'Cpu_Brand_Other Intel Processors': [1 if company == 'Other Intel Processors' else 0],
        'Gpu_brand_AMD': [1 if company == 'AMD' else 0],
        'Gpu_brand_Intel': [1 if company == 'Intel' else 0],
        'Gpu_brand_Nvidia': [1 if company == 'Nvidia' else 0],
        'os_Mac': [1 if company == 'Mac' else 0],
        'os_Others/No OS/Linux': [1 if company == 'Others/No OS/Linux' else 0],
        'os_Windows': [1 if company == 'Windows' else 0]
    }, index=[0])
    if st.button('Predict Price'):
        pred = model.predict(data_new)
        pred_exp = np.exp(pred[0])
        st.success('Prediction is {:.2f}'.format(pred_exp))

if __name__ == '__main__':
    main()