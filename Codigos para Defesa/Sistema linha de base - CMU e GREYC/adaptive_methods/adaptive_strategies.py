
import pandas

def update(strategy, detector, biometric_reference, new_features):
    if strategy == 'GrowingWindow':
        new_model = GrowingWindow(detector, biometric_reference, new_features)
    elif strategy == 'SlidingWindow':
        new_model = SlidingWindow(detector, biometric_reference, new_features)
    elif strategy == 'DoubleParallel':
        new_model = DoubleParallel(detector, biometric_reference, new_features)
    else:
        print(strategy)
        raise Exception(f"Escolha uma estratégia de adaptação válida! {strategy}")
    return new_model

def GrowingWindow(detector, biometric_reference, new_features):
    biometric_reference.features = pandas.concat([biometric_reference.features, pandas.DataFrame(new_features).T], ignore_index=True)
    new_model = detector.train(training_data=biometric_reference.features)
    return new_model

def SlidingWindow(detector, biometric_reference, new_features):
    biometric_reference.features = biometric_reference.features.iloc[1:]
    biometric_reference.features = pandas.concat([biometric_reference.features, pandas.DataFrame(new_features).T], ignore_index=True)
    new_model = detector.train(training_data = biometric_reference.features)
    return new_model

def DoubleParallel(detector, biometric_reference, new_features):
    gw_model = GrowingWindow(detector, biometric_reference[0], new_features)
    sw_model = SlidingWindow(detector, biometric_reference[1], new_features)
    return gw_model, sw_model

