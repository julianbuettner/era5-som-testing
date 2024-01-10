#!/usr/bin/env python3

from datetime import datetime, timedelta

GTD_FILE = "GTD_labelling_ERA5_JJAextd_1979-2019.csv"
DATE_FORMAT = "%d-%m-%Y"
XARRAY_DATE_FORMAT = "%Y-%m-%d"

def is_num(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_bes():
    result = []
    with open(GTD_FILE, "r") as f:
        for line in f.readlines():
            if not is_num(line[:2]):
                continue
            first, is_som = line.split(",")[:2]
            if is_som in ["0"]:
                continue
            elif is_som in ["1"]:
                pass
            else:
                raise ValueError("Bool should be 1 or 0")
            (from_date, _ignore, to_date) = first.split(" ")
            from_date = datetime.strptime(from_date, DATE_FORMAT)
            to_date = datetime.strptime(to_date, DATE_FORMAT)
            while from_date <= to_date:
                insert_str = from_date.strftime(XARRAY_DATE_FORMAT)
                if not insert_str in result:
                    result.append(insert_str)
                from_date += timedelta(days=1)
    return result

if __name__ == '__main__':
    bis = get_bes()
    print(f"Found {len(bis)} BEs")
    print(bis[:15])
