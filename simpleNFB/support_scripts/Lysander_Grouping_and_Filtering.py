# Meta Data Comparison

def has_data_set_same_meta_data(d):

    """
    Checks if all data sets have the same meta data. i.e. the same current, bias, exposure time, grating density, center wavelength and tip position.
    
    Input:
    --------
    d: list of import datafiles from stml_dat_data_loader

    Output:
    --------
    all_current_values_are_the_same: bool
    all_bias_values_are_the_same: bool
    all_exposure_time_values_are_the_same: bool
    all_grating_density_values_are_the_same: bool
    all_center_wavelength_values_are_the_same: bool
    all_tip_positions_are_the_same: bool
    """

    # Do all files for the plotting have the same current?
    all_currents = [round(float(d[i].current[0]/10))*10 for i in range(len(d))]
    if len(set(all_currents)) == 1:
        all_current_values_are_the_same = True
    else:
        all_current_values_are_the_same = False

    # Do all files for the plotting have the same bias?
    all_biases = [d[i].bias[0] for i in range(len(d))]
    if len(set(all_biases)) == 1:
        all_bias_values_are_the_same = True
    else:
        all_bias_values_are_the_same = False

    # Do all files for the plotting have the same exposure time?
    all_exposure_times = [d[i].exposure_time[0] for i in range(len(d))]
    if len(set(all_exposure_times)) == 1:
        all_exposure_time_values_are_the_same = True
    else:
        all_exposure_time_values_are_the_same = False

    # Do all files for the plotting have the same grating density?
    all_grating_densities = [d[i].grating_density[0] for i in range(len(d))]
    if len(set(all_grating_densities)) == 1:
        all_grating_density_values_are_the_same = True
    else:
        all_grating_density_values_are_the_same = False

    # Do all files for the plotting have the same center wavelength?
    all_center_wavelengths = [d[i].center_wavelength[0] for i in range(len(d))]
    if len(set(all_center_wavelengths)) == 1:
        all_center_wavelength_values_are_the_same = True
    else:
        all_center_wavelength_values_are_the_same = False

    # Are all files for the plotting recorded at the same position?
    all_tip_x_positions = [float(d[i].tip_x[0]) for i in range(len(d))]
    all_tip_y_positions = [float(d[i].tip_y[0]) for i in range(len(d))]
    if len(set(all_tip_x_positions)) == 1 and len(set(all_tip_y_positions)) == 1:
        all_tip_positions_are_the_same = True
    else:
        all_tip_positions_are_the_same = False

    return all_current_values_are_the_same, all_bias_values_are_the_same, all_exposure_time_values_are_the_same, all_grating_density_values_are_the_same, all_center_wavelength_values_are_the_same, all_tip_positions_are_the_same

### Grouping STML Spectra with the same properties (current,bias,exposure_time,grating_density,center_wavelength,tip_positions) and enforce meta data exceptions ###

def group_stml_data_by_meta_data(stml_data, prop_dict):
    """
    Groups stml_data objects by their meta data properties, considering exception flags.

    Args:
    stml_data (list): List of stml_data_loader objects
    prop_dict (dict): Dictionary containing exception flags and thresholds

    Returns:
    meta_property_groups (list): List of groups of stml_data indexes with the same meta data
    """

    # Grouping data sets with the same meta data
    used_file_indexes = []
    meta_property_groups = []
    for i in range(len(stml_data)):
        if i not in used_file_indexes:
            group = [i]
            used_file_indexes.append(i)
            for j in range(len(stml_data)):
                if i !=j and j not in used_file_indexes:
                    b1,b2,b3,b4,b5,b6 = has_data_set_same_meta_data([stml_data[i],stml_data[j]])
                    if prop_dict['force_current_to_be_same']:
                        b1 = True
                    if prop_dict['force_location_to_be_same']:
                        b6 = True
                    if prop_dict['current_similarity_threshold_nA'] != False:
                        current_stml_data1 = abs(round(float(stml_data[i].current[0]/10))*10)
                        current_stml_data2 = abs(round(float(stml_data[j].current[0]/10))*10)
                        if abs(current_stml_data1-current_stml_data2)/1000 <= prop_dict['current_similarity_threshold_nA']:
                            b1 = True
                    if b1 and b2 and b3 and b4 and b5 and b6 == True:
                        group.append(j)
                        used_file_indexes.append(j)
            meta_property_groups.append(group)

    return meta_property_groups

def cosmic_ray_group_comparison(stml_data, group_threshold):

    """
    Identifies and removes outliers in a group of three stml_data_loader objects by comparing their intensities and using a majority voting scheme.

    Args:
    stml_data (list): List of three stml_data_loader objects
    group_threshold (float): Threshold for intensity difference to consider data points as consistent

    Returns:
    None: The function modifies the input stml_data objects in place by setting outlier intensities to NaN.
    """

    # stml_data needs to be a list of three stml_dat_data_loader objects
    assert len(stml_data) == 3, "cosmic_ray_group_comparision: Input data must be a list of three stml_dat_data_loader objects"

    # Extract wavelengths and all intensities
    wavelengths = stml_data[0].wavelength
    all_intensities = np.array([data.intensity for data in stml_data])

    # Initialize majority voting and permutation votes arrays
    num_samples, num_points = all_intensities.shape
    permutations = list(itertools.combinations(range(num_samples), 2))
    permutations_votes = np.zeros((len(permutations), num_points))

    # Compute votes for each permutation
    for i in range(num_points):
        for idx, (j, k) in enumerate(permutations):
            if np.isnan(all_intensities[j, i]) or np.isnan(all_intensities[k, i]):
                permutations_votes[idx, i] = 1
            elif abs(all_intensities[j, i] - all_intensities[k, i]) < group_threshold:
                permutations_votes[idx, i] = 1

    # Identify outliers and update data
    detected_outliers_wavelength = []
    nan_data_points = np.zeros(all_intensities.shape, dtype=bool)

    for i in range(num_points):
        votes = permutations_votes[:, i]
        if np.mean(votes) < 1:
            if np.sum(votes) == 1:
                outlier_perm_idx = np.where(votes == 1)[0][0]
                outlier_sample_idx = [k for k in range(num_samples) if k not in permutations[outlier_perm_idx]]
                nan_data_points[outlier_sample_idx[0], i] = True
            elif np.sum(votes) == 0:
                detected_outliers_wavelength.append(wavelengths[i])

    # Apply NaN to identified outliers
    for i in range(num_samples):
        stml_data[i].intensity[nan_data_points[i]] = np.nan

    # Debugging output
    # print("Detected outliers wavelengths:", detected_outliers_wavelength)
    # print("NaN data points:", np.argwhere(nan_data_points))

    return stml_data, detected_outliers_wavelength




### How to use:

# # Outlier removal by comparision within group
# if outlier_removal_by_group_comparision_active:
#     for group in meta_property_groups:
#         if len(group) !=3:
#             print('Error: Group size is not 3 with following filenames')
#             # Print Filenames and FileIDXs
#             for i in group:
#                 print('FileIDX: '+str(i)+' Filename: '+stml_data[i].filename)
#             continue
#         else:
#             stml_data_group = [stml_data[i] for i in group]
#             stml_data_group, detected_outliers_wavelength = cosmic_ray_group_comparison(stml_data_group, outlier_removal_by_group_comparision_active_threshold)

#             # Adding process information to the data sets
#             group_filenames = []
#             for i in range(len(stml_data_group)):
#                 group_filenames.append(stml_data_group[i].filename)
#             for i in range(len(stml_data_group)):
#                 stml_data_group[i].additional_information_container['outlier_removal_by_group_comparision_active'] = True
#                 stml_data_group[i].additional_information_container['outlier_removal_by_group_comparision_active_threshold'] = outlier_removal_by_group_comparision_active_threshold
#                 stml_data_group[i].additional_information_container['detected_outliers_wavelength'] = detected_outliers_wavelength
#                 stml_data_group[i].additional_information_container['group_filenames'] = group_filenames
            
#             # Replace the data sets in the stml_data list with the processed data sets
#             for i in range(len(group)):
#                 stml_data[group[i]] = stml_data_group[i]