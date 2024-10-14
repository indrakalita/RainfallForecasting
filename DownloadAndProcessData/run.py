import os
import datetime
import functions


def process_variables(source_parent, output_parent, vname, first, levels=None, mask=False, start=0):
    nc_files = functions.get_files(source_parent)

    # Get dataset information
    functions.print_info(nc_files)

    functions.count_masked_variable(nc_files, vname)

    time = functions.get_time(first, nc_files)
    #print(time)

    varr = functions.extract_var_array(nc_files, vname, start, use_mask=mask)
    print(varr.shape)

    #varr_resized = functions_.resize(varr, vname, (levels is not None))

    functions.save_to_daily_files(output_parent, time, varr, vname, resized=False, use_mask=mask, levels=levels)

    pass


if __name__ == "__main__":
    source = "path_to_downloaded_nc_files"  # change as needed
    output = "path_to_output_destination"   # change as needed
    firstDay = datetime.datetime(1900, 1, 1, hour=0, minute=0, second=0)  # time units for ERA5 are hours since 1/1/1990

    levels = ['300', '500', '600', '700', '850', '925', '950']
    levels_variables = ["crwc", "q", "r", "t", "u", "v", "w"]
    single_variables = ["cape", "cin", "d2m", "kx", "sp", "t2m", "tcc", "tclw", "tcwv", "vimd"]

    for var in levels_variables:
        print(f"Processing variable {var} on pressure levels.")
        process_variables(source + var + "/", output, var, firstDay, levels=levels, mask=False, start=0)

    for var in single_variables:
        print(f"Processing variable {var} on single levels.")
        process_variables(source + var + "/", output, var, firstDay, mask=False, start=0)
