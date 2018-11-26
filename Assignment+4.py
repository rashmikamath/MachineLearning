
# coding: utf-8

# In[ ]:

def blight_model():
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler

    # Load train and test data
    train_data = pd.read_csv('train.csv', encoding = 'ISO-8859-1')
    test_data = pd.read_csv('test.csv')

    # Filter null valued compliance rows
    train_data = train_data[(train_data['compliance'] == 0) | (train_data['compliance'] == 1)]
    address =  pd.read_csv('addresses.csv')

    # Load address and location information
    latlons = pd.read_csv('latlons.csv')
    address = address.set_index('address').join(latlons.set_index('address'), how='left')

    # Join address and location to train and test data
    train_data = train_data.set_index('ticket_id').join(address.set_index('ticket_id'))
    test_data = test_data.set_index('ticket_id').join(address.set_index('ticket_id'))

    # Filter null valued hearing date rows
    train_data = train_data[~train_data['hearing_date'].isnull()]

    # Remove Non Existing Features In Test Data
    train_remove_list = [
            'balance_due',
            'collection_status',
            'compliance_detail',
            'payment_amount',
            'payment_date',
            'payment_status'
        ]

    train_data.drop(train_remove_list, axis=1, inplace=True)

    # Remove String Data
    string_remove_list = ['violator_name', 'zip_code', 'country', 'city',
            'inspector_name', 'violation_street_number', 'violation_street_name',
            'violation_zip_code', 'violation_description',
            'mailing_address_str_number', 'mailing_address_str_name',
            'non_us_str_code', 'agency_name', 'state', 'disposition',
            'ticket_issued_date', 'hearing_date', 'grafitti_status', 'violation_code'
        ]

    train_data.drop(string_remove_list, axis=1, inplace=True)
    test_data.drop(string_remove_list, axis=1, inplace=True)

    # Fill NA Lat Lon Values
    train_data.lat.fillna(method='pad', inplace=True)
    train_data.lon.fillna(method='pad', inplace=True)
    test_data.lat.fillna(method='pad', inplace=True)
    test_data.lon.fillna(method='pad', inplace=True)

    # Select target value as y train and remove it from x train
    y_train = train_data.compliance
    X_train = train_data.drop('compliance', axis=1)

    # Do nothing with test data and select as x test, we don't have y_test
    X_test = test_data
    
    # Scale Features To Reduce Computation Time
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build And Train Classifier Model
    clf = MLPClassifier(hidden_layer_sizes = [100, 10],
                        alpha=0.001,
                        random_state = 0, 
                        solver='lbfgs', 
                        verbose=0)
    clf.fit(X_train_scaled, y_train)
    
    # Predict probabilities
    y_proba = clf.predict_proba(X_test_scaled)[:,1]
    
    # Integrate with reloaded test data
    test_df = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    test_df['compliance'] = y_proba
    test_df.set_index('ticket_id', inplace=True)
    
    return test_df.compliance 

