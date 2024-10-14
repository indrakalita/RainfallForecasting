import os
import datetime
import netCDF4 as nc
import numpy as np


def get_files(topdir):
    flist = []

    if len(topdir) > 0:
        files = os.listdir(topdir)
        files.sort()

        for f in files:
            flist.append(topdir + f)

    return flist


def print_info(files):
    nc_obj = nc.Dataset(files[0])
    print("Creating data object from .nc file:")
    print(nc_obj)
    print(nc_obj.variables)
    pass


def count_masked_variable(files, variable):
    for f in files:
        nc_obj = nc.Dataset(f)
        var = nc_obj[variable][:]

        # Keep just the ERA5 version, which is index 0 in the 2nd dimension
        if len(var.shape) > 3:
            var = var[:, 0, :, :]

        masked = np.ma.count_masked(var)
        total = var.size
        print("{num} masked values in file {file} along {v}".format(num=masked, file=f, v=variable))
        print("{tot} total values in file {file} along {v}".format(tot=total, file=f, v=variable))
    pass


def get_time(first_date, files):
    time_vec = []

    for f in files:
        nc_obj = nc.Dataset(f)
        time = nc_obj.variables["time"][:] 

        for t in time:
            time_vec.append(datetime.timedelta(hours=int(t)) + first_date)

    return time_vec


def unmask(arr, use_mean=True, new_val=None):
    mask = arr.mask

    if use_mean:
        mean = np.mean(arr)
        arr[mask] = mean
    else:
        arr[mask] = new_val

    return arr.data


def _extract_array_from_nc(file, vname, mask):
    nc_obj = nc.Dataset(file)
    var = nc_obj.variables[vname][:]

    if not mask and type(var) == np.ma.masked_array:
        var = unmask(var)

    return var


def extract_var_array(files, vname, start=0, use_mask=False):
    var = _extract_array_from_nc(files[0], vname, mask=use_mask)

    for i in range(len(files)):
        if i == 0:
            continue
        else:
            curr_arr = _extract_array_from_nc(files[i], vname, mask=use_mask)
            var = np.vstack((var, curr_arr))

    return var #[start:, ]


def scale(arr, factor):
    scaled = arr * factor
    return scaled


def resize(arr, vname, levels=False):
    print(f"Resizing {vname}.")

    resized = arr.copy()

    if len(arr.shape) == 3 and not levels:  # non-wind, non-leveled variables
        temp1 = np.repeat(arr, 2, axis=1)
        temp2 = np.repeat(temp1, 3, axis=2)
        temp3 = np.pad(temp2, pad_width=((0, 0), (2, 2), (0, 0)), mode="edge")
        resized = temp3[:, :, 0:64]

    else:  # wind and leveled variables
        temp1 = np.repeat(arr, 2, axis=2)
        temp2 = np.repeat(temp1, 3, axis=3)
        temp3 = np.pad(temp2, pad_width=((0, 0), (0, 0), (2, 2), (0, 0)), mode="edge")
        resized = temp3[:, :, :, 0:64]

    print(f"{vname} new shape: {resized.shape}")
    return resized


def compute_laplacian(var_array, vname, loc):

    assert(len(loc) == 2)

    print("Computing Laplacian for variable " + vname)
    var_shape = np.array(var_array.shape)
    row_border = var_shape[-2]-1  # 29 or 63
    col_border = var_shape[-1]-1  # 22 or 63

    try:
        center = var_array[:, loc[0], loc[1]]
    except IndexError:
        print("The chosen location is out of bounds, please choose indices within " + str(var_shape[1:]))
        loc = [int(x) for x in input("New location (space between integer indices): ").split()]
        center = var_array[:, loc[0], loc[1]]

    north, south, east, west = None, None, None, None

    # not on a corner/edge:
    if (loc[0] not in [0, row_border]) and (loc[1] not in [0, col_border]):
        north = var_array[:, loc[0] - 1, loc[1]]
        south = var_array[:, loc[0] + 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]
        west = var_array[:, loc[0], loc[1] - 1]

    # northwest corner
    elif (loc[0] == 0) and (loc[1] == 0):
        south = var_array[:, loc[0] + 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]

    # northeast corner
    elif (loc[0] == row_border) and (loc[1] == 0):
        south = var_array[:, loc[0] + 1, loc[1]]
        west = var_array[:, loc[0], loc[1] - 1]

    # southwest corner
    elif (loc[0] == 0) and [loc[1] == col_border]:
        north = var_array[:, loc[0] - 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]

    # southeast corner
    elif (loc[0] == row_border) and (loc[1] == col_border):
        north = var_array[:, loc[0] - 1, loc[1]]
        west = var_array[:, loc[0], loc[1] - 1]

    # north edge
    elif loc[0] == 0:
        south = var_array[:, loc[0] + 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]
        west = var_array[:, loc[0], loc[1] - 1]

    # south edge
    elif loc[0] == col_border:
        north = var_array[:, loc[0] - 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]
        west = var_array[:, loc[0], loc[1] - 1]

    # east edge
    elif loc[1] == row_border:
        north = var_array[:, loc[0] - 1, loc[1]]
        south = var_array[:, loc[0] + 1, loc[1]]
        west = var_array[:, loc[0], loc[1] - 1]

    # west edge
    elif loc[1] == 0:
        north = var_array[:, loc[0] - 1, loc[1]]
        south = var_array[:, loc[0] + 1, loc[1]]
        east = var_array[:, loc[0], loc[1] + 1]

    # out of bounds
    else:
        raise IndexError("The chosen location is out of bounds, please choose indices within " + var_shape[-2, -1])

    laplace = center - np.mean(np.array([north, south, east, west]), axis=0)
    return laplace


def get_level(arr, level):
    assert arr.shape > 3
    return arr[:, level, :, :]


def _save_to_daily_files(path, time, arr, fname):
    date_min = time[0]
    current_date = date_min
    i = 0

    while i < len(arr):
        fname_day = fname
        day_arr = arr[i:i + 4, :, :]
        fname_day += current_date.strftime("%Y%m%d")

        np.save(path + fname_day, day_arr)
        i += 4
        current_date = current_date + datetime.timedelta(days=1)
    pass


def save_to_daily_files(outdir, time, arr, vname, resized=False, use_mask=False, levels=None):

    # Save arrays for variables at levels in new folder for each level
    if levels is not None:
        assert len(levels) == arr.shape[1]

        for l in range(len(levels)):
            vname_l = vname
            vname_l += str(levels[l])
            parent_dir = outdir + vname_l

            if resized:
                parent_dir += "_resized"

            try:
                os.mkdir(parent_dir)
            except FileExistsError:
                print("Variable directory already exists.")
                
            print(f"Saving daily arrays to files in {parent_dir}")

            level_arr = arr[:, l, :, :]

            fname = vname_l + "_"
            if resized:
                fname += "resized_"

            _save_to_daily_files(parent_dir + "/", time, level_arr, fname)

    else:
        parent_dir = outdir + vname

        if resized:
            parent_dir += "_resized"
        try:
            os.mkdir(parent_dir)
        except FileExistsError:
            print("Variable directory already exists.")
            
        print(f"Saving daily arrays to files in {parent_dir}")

        fname = vname
        if resized:
            fname += "_resized"

        if use_mask:
            fname += "_masked_"
        else:
            fname += "_filled_"

        _save_to_daily_files(parent_dir + "/", time, arr, fname)
    pass
