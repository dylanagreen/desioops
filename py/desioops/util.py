from astropy import units as u
from astropy.time import Time, TimeDelta

def filename_to_time_str(fname, noshift=False):
    fname = fname.replace(".jpg", "")
    # TODO docstring
    # At least with files in this format and 24 hour time the timestamp/filename
    # is a monotonically increasing integer. On this date/time
    # they switched from saving files in local time to the utc time
    # as the file name. The UTC of the filename is not exactly the time
    # the image was saved, but is close enough for our purposes (it's only off
    # by a minute at most)
    if int(fname) < 20260216124936:
       shift = TimeDelta(7 * u.hour)
    else:
        shift = TimeDelta(0 * u.hour)

    year = fname[:4]
    month = fname[4:6]
    day = fname[6:8]

    hour = fname[8:10]
    minute = fname[10:12]
    second = fname[12:14]
    tstring = f"{year}-{month}-{day}T{hour}:{minute}:{second}"

    if noshift:
        return tstring
    return tstring, shift

def filename_to_time(fname, noshift=False):
    tstring, shift = filename_to_time_str(fname, noshift)
    return Time(tstring, scale="utc") + shift