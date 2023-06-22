# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:27:14 2023

@author: carlo
"""

import pandas as pd
import numpy as np
from decimal import Decimal
import re, time, sys, os

from datetime import datetime, timedelta, date, time
import time
from pytz import timezone
import doctest
import config_carmax

import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
output_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(output_path) # current working directory

def combine_specs(w, ar, d, mode = 'SKU'):
    '''
    
    Parameters
    - w: string
        section_width
    - ar: string
        aspect_ratio
    - d: string
        diameter
    - mode: string; optional
        SKU or MATCH
    
    Returns
    - combined specs with format for SKU or matching
    
    >>> combine_specs('175', 'R', 'R15', mode = 'SKU')
    '175/R15'
    >>> combine_specs('175', '65', 'R15', mode = 'SKU')
    '175/65/R15'
    >>> combine_specs('33', '12.5', 'R15', mode = 'SKU')
    '33X12.5/R15'
    >>> combine_specs('LT175', '65', 'R15C', mode = 'SKU')
    'LT175/65/R15C'
    >>> combine_specs('LT175', '65', 'R15C', mode = 'MATCH')
    '175/65/15'
    >>> combine_specs('175', '65', '15', mode = 'SKU')
    '175/65/R15'
    
    '''
    
    if mode == 'SKU':
        d = d if 'R' in d else 'R' + d 
        if ar != 'R':
            if '.' in ar:
                return w + 'X' + ar + '/' + d
            else:
                return '/'.join([w, ar, d])
        else:
            return w + '/' + d
            
    elif mode == 'MATCH':
        w = ''.join(re.findall('[0-9]|\.', str(w)))
        ar = ''.join(re.findall('[0-9]|\.|R', str(ar)))
        d = ''.join(re.findall('[0-9]|\.', str(d)))
        return '/'.join([w, ar, d])

    else:
        combine_specs(str(w), str(ar), str(d), mode = 'SKU')

def fix_names(sku_name, comp=None):
    '''
    Fix product names to match competitor names
    
    Parameters
    ----------
    sku_name: str
        input SKU name string
    comp: list (optional)
        optional list of model names to compare with
    
    Returns
    -------
    name: str
        fixed names as UPPERCASE
    '''
    
    # replacement should be all caps
    change_name_dict = {'TRANSIT.*ARZ.?6-X' : 'TRANSITO ARZ6-X',
                        'TRANSIT.*ARZ.?6-A' : 'TRANSITO ARZ6-A',
                        'TRANSIT.*ARZ.?6-M' : 'TRANSITO ARZ6-M',
                        'OPA25': 'OPEN COUNTRY A25',
                        'OPA28': 'OPEN COUNTRY A28',
                        'OPA32': 'OPEN COUNTRY A32',
                        'OPA33': 'OPEN COUNTRY A33',
                        'OPAT\+': 'OPEN COUNTRY AT PLUS', 
                        'OPAT2': 'OPEN COUNTRY AT 2',
                        'OPMT2': 'OPEN COUNTRY MT 2',
                        'OPAT OPMT': 'OPEN COUNTRY AT',
                        'OPAT': 'OPEN COUNTRY AT',
                        'OPMT': 'OPEN COUNTRY MT',
                        'OPRT': 'OPEN COUNTRY RT',
                        'OPUT': 'OPEN COUNTRY UT',
                        'DC -80': 'DC-80',
                        'DC -80+': 'DC-80+',
                        'KM3': 'MUD-TERRAIN T/A KM3',
                        'KO2': 'ALL-TERRAIN T/A KO2',
                        'TRAIL-TERRAIN T/A' : 'TRAIL-TERRAIN',
                        '265/70/R16 GEOLANDAR 112S': 'GEOLANDAR A/T G015',
                        '265/65/R17 GEOLANDAR 112S' : 'GEOLANDAR A/T G015',
                        '265/65/R17 GEOLANDAR 112H' : 'GEOLANDAR G902',
                        'GEOLANDAR A/T 102S': 'GEOLANDAR A/T-S G012',
                        'GEOLANDAR A/T': 'GEOLANDAR A/T G015',
                        'ASSURACE MAXGUARD SUV': 'ASSURANCE MAXGUARD SUV',
                        'EFFICIENTGRIP SUV': 'EFFICIENTGRIP SUV',
                        'EFFICIENGRIP PERFORMANCE SUV':'EFFICIENTGRIP PERFORMANCE SUV',
                        'WRANGLE DURATRAC': 'WRANGLER DURATRAC',
                        'WRANGLE AT ADVENTURE': 'WRANGLER AT ADVENTURE',
                        'WRANGLER AT ADVENTURE': 'WRANGLER AT ADVENTURE',
                        'WRANGLER AT SILENT TRAC': 'WRANGLER AT SILENTTRAC',
                        'ENASAVE  EC300+': 'ENSAVE EC300 PLUS',
                        'SAHARA AT2' : 'SAHARA AT 2',
                        'SAHARA MT2' : 'SAHARA MT 2',
                        'WRANGLER AT SILENT TRAC': 'WRANGLER AT SILENTTRAC',
                        'POTENZA RE003 ADREANALIN': 'POTENZA RE003 ADRENALIN',
                        'POTENZA RE004': 'POTENZA RE004',
                        'SPORT MAXX 050' : 'SPORT MAXX 050',
                        'DUELER H/T 470': 'DUELER H/T 470',
                        'DUELER H/T 687': 'DUELER H/T 687 RBT',
                        'DUELER A/T 697': 'DUELER A/T 697',
                        'DUELER A/T 693': 'DUELER A/T 693 RBT',
                        'DUELER H/T 840' : 'DUELER H/T 840 RBT',
                        'EVOLUTION MT': 'EVOLUTION M/T',
                        'BLUEARTH AE61' : 'BLUEARTH XT AE61',
                        'BLUEARTH ES32' : 'BLUEARTH ES ES32',
                        'BLUEARTH AE51': 'BLUEARTH GT AE51',
                        'COOPER STT PRO': 'STT PRO',
                        'COOPER AT3 LT' : 'AT3 LT',
                        'COOPER AT3 XLT' : 'AT3 XLT',
                        'A/T3' : 'AT3',
                        'ENERGY XM+' : 'ENERGY XM2+',
                        'XM2+' : 'ENERGY XM2+',
                        'AT3 XLT': 'AT3 XLT',
                        'ADVANTAGE T/A DRIVE' : 'ADVANTAGE T/A DRIVE',
                        'ADVANTAGE T/A SUV' : 'ADVANTAGE T/A SUV'
                        }
    
    if pd.isna(sku_name) or (sku_name is None):
        return np.NaN
    
    else:
        # uppercase and remove double spaces
        raw_name = re.sub('  ', ' ', sku_name).upper().strip()
        # specific cases
        for key in change_name_dict.keys():
            if re.search(key, raw_name):
                return change_name_dict[key]
            else:
                continue
        
        # if match list provided
        
        if comp is not None:
            # check if any name from list matches anything in sku name
            match_list = [n for n in comp if re.search(n, raw_name)]
            # exact match from list
            if len(match_list) == 1:
                return match_list[0]
            # multiple matches (i.e. contains name but with extensions)
            elif len(match_list) > 1:
                long_match = ''
                for m in match_list:
                    if len(m[0]) > len(long_match):
                        long_match = m[0]
                return long_match
            # no match
            else:
                return raw_name
        else:
            return raw_name
    

def remove_trailing_zero(num):
    '''
    Removes unnecessary zeros from decimals

    Parameters
    ----------
    num : Decimal(number)
        number applied with Decimal function (see import decimal from Decimal)

    Returns
    -------
    number: Decimal
        Fixed number in Decimal form

    '''
    return num.to_integral() if num == num.to_integral() else num.normalize()

def clean_width(w, model = None):
    '''
    Clean width values
    
    Parameters
    ----------
    d: string
        width values in string format
        
    Returns:
    --------
    d: string
        cleaned diameter values
    
    DOCTESTS:
    >>> clean_width('7')
    '7'
    >>> clean_width('175')
    '175'
    >>> clean_width('6.50')
    '6.5'
    >>> clean_width('27X')
    '27'
    >>> clean_width('LT35X')
    'LT35'
    >>> clean_width('8.25')
    '8.25'
    >>> clean_width('P265.5')
    'P265.5'
    >>> clean_width(np.NaN)
    nan
    
    '''
    if pd.notna(w):
        w = str(w).strip().upper()
        # detects if input has expected format
        prefix_num = re.search('[A-Z]*[0-9]+.?[0-9]*', w)
        if prefix_num is not None:
            num_str = ''.join(re.findall('[0-9]|\.', prefix_num[0]))
            num = str(remove_trailing_zero(Decimal(num_str)))
            prefix = w.split(num_str)[0]
            return prefix + num
        else:
            return np.NaN
    else:
        if model is None:
            return np.NaN
        else:
            try:
                width = model.split('/')[0].split(' ')[-1].strip().upper()
                return clean_width(width)   
            except:
                return np.NaN
    
def clean_diameter(d):
    '''
    Fix diameter values
    
    Parameters
    ----------
    d: string
        diameter values in string format
        
    Returns:
    --------
    d: string
        fixed diameter values
    
    DOCTESTS:
    >>> clean_diameter('R17LT')
    'R17LT'
    >>> clean_diameter('R22.50')
    'R22.5'
    >>> clean_diameter('15')
    'R15'
    >>> clean_diameter(np.NaN)
    nan
    
    '''
    if pd.notna(d):
        d = str(d).strip().upper()
        num_suffix = re.search('[0-9]+.?[0-9]*[A-Z]*', d)
        if num_suffix is not None:
            num_str = ''.join(re.findall('([0-9]|\.)', num_suffix[0]))
            num = str(remove_trailing_zero(Decimal(num_str)))
            suffix = num_suffix[0].split(num_str)[-1]
            return f'R{num}{suffix}'
    else:
        return np.NaN

def clean_aspect_ratio(ar, model = None):
    
    '''
    Clean raw aspect ratio input
    
    Parameters
    ----------
    ar: float or string
        input raw aspect ratio data
    model: string; optional
        input model string value of product
        
    Returns
    -------
    ar: string
        fixed aspect ratio data in string format for combine_specs
    
    DOCTESTS:
    >>> clean_aspect_ratio('/')
    'R'
    >>> clean_aspect_ratio('.5')
    '9.5'
    >>> clean_aspect_ratio('14.50')
    '14.5'
    >>> clean_aspect_ratio(np.NaN)
    'R'
    
    '''
    error_ar = {'.5' : '9.5',
                '0.': '10.5',
                '2.': '12.5',
                '3.': '13.5',
                '5.': '15.5',
                '70.5': '10.5'}
    
    if pd.notna(ar):
        # aspect ratio is faulty
        if str(ar) in ['0', 'R1', '/', 'R']:
            return 'R'
        # incorrect parsing osf decimal aspect ratios
        elif str(ar) in error_ar.keys():
            return error_ar[str(ar)]
        # numeric/integer aspect ratio
        elif str(ar).isnumeric():
            return str(ar)
        # alphanumeric
        elif str(ar).isalnum():
            return ''.join(re.findall('[0-9]', str(ar)))
        
        # decimal aspect ratio with trailing 0
        elif '.' in str(ar):
            return str(remove_trailing_zero(Decimal(str(ar))))
        
        else:
            return np.NaN
        
    else:
        return 'R'

def clean_speed_rating(sp):
    '''
    Clean speed rating of gulong products
    
    DOCTESTS:
    >>> clean_speed_rating('W XL')
    'W'
    >>> clean_speed_rating('0')
    'B'
    >>> clean_speed_rating('118Q')
    'Q'
    >>> clean_speed_rating('T/H')
    'T'
    >>> clean_speed_rating('-')
    nan
    
    '''
    # SAILUN 205/75/R16C COMMERCIO VX1 10PR - 113/111R
    # SAILUN 205/75/R16C COMMERCIO VX1 8PR - 110/108R
    # SAILUN 235/65/R16C COMMERCIO VX1 8PR - 115/113R
    # SAILUN 33X/12.50/R20 TERRAMAX M/T 10PR - 114Q
    # SAILUN 35X/12.50/R20 TERRAMAX M/T 10PR - 121Q
    # SAILUN 305/55/R20 TERRAMAX M/T 10PR - 121/118Q
    # SAILUN 35X/12.50/R18 TERRAMAX M/T 10PR - None
    # SAILUN 33X/12.50/R18 TERRAMAX M/T 10PR - 118Q
    # SAILUN 35X/12.50/R17 TERRAMAX M/T 10PR - 121Q
    # SAILUN 33X/12.50/R17 TERRAMAX M/T 8PR - 114Q
    # SAILUN 285/70/R17 TERRAMAX M/T 10PR - 121/118Q
    # SAILUN 265/70/R17 TERRAMAX M/T 10PR - 121/118Q
    # SAILUN 265/75/R16 TERRAMAX M/T 10PR - 116S
    # SAILUN 245/75/R16 TERRAMAX M/T 10PR - 111S
    # SAILUN 35X/12.50/R15 TERRAMAX M/T 6PR - 113Q
    # SAILUN 33X/12.50/R15 TERRAMAX M/T 6PR - 108Q
    # SAILUN 31X/10.50/R15 TERRAMAX M/T 6PR - 109S
    # SAILUN 30X/9.50/R15 TERRAMAX M/T 6PR - 104Q
    # SAILUN 235/75/R15 TERRAMAX M/T 6PR - 104/101Q
    # SAILUN 265/70/R17 TERRAMAX A/T 10PR - 121/118S
    
    # not NaN
    if pd.notna(sp):
        # baseline correct
        sp = sp.strip().upper()
        # detect if numerals are present 
        num = re.search('[0-9]{2,3}', sp)
        
        if num is None:
            pass
        else:
            # remove if found
            sp = sp.split(num[0])[-1].strip()
            
        if 'XL' in sp:
            return sp.split('XL')[0].strip()
        elif '/' in sp:
            return sp.split('/')[0].strip()
        elif sp == '0':
            return 'B'
        elif sp == '-':
            return np.NaN
        else:
            return sp
    else:
        return np.NaN

def raw_specs(x):
    if (str(x['aspect_ratio']) == 'nan' or x['aspect_ratio'] == 0) and str(x['diameter'])[-1] != 'C':
        return '/'.join([str(x['width']), str(x['diameter'])+'C'])
    elif (str(x['aspect_ratio']) == 'nan' or x['aspect_ratio'] == 0) and str(x['width'])[-1] == 'C':
        return '/'.join([str(x['width']), str(x['diameter'])])
    elif str(x['aspect_ratio']) == '/':
        return '/'.join([str(x['width']), str(x['diameter'])])
    else:
        return '/'.join([str(x['width']), str(x['aspect_ratio']), str(x['diameter'])])

def combine_sku(make, w, ar, d, model, load, speed):
    '''
    DOCTESTS:
            
    >>> combine_sku('ARIVO', '195', 'R', 'R15', 'TRANSITO ARZ 6-X', '106/104', 'Q')
    'ARIVO 195/R15 TRANSITO ARZ 6-X 106/104Q'
    
    '''
    specs = combine_specs(w, ar, d, mode = 'SKU')
    
    if (load in ['nan', np.NaN, None]) or (speed in ['nan', np.NaN, None]):
        return ' '.join([make, specs, model])
    else:
        return ' '.join([make, specs, model, load + speed])

def promo_GP(price, cost, sale_tag, promo_tag):
    '''
    price : float
        price_gulong price
    cost : float
        supplier cost price
    sale_tag : binary
        buy 4 tires 3% off per tire
    promo_tag : binary
        buy 3 tires get 1 free
        
    DOCTESTS:
    >>> promo_GP(4620, 3053.65, 1, 0)
    5711
    >>> promo_GP(5040, 3218.4, 0, 1)
    2246.4
        
    '''
    # sale tag : buy 4 tires 3% off per tire
    # promo tag: buy 3 tires get 1 free
    if sale_tag:
        gp = (price * 0.97 - cost) * 4
    else:
        gp = (price * 3 - cost * 4)
    return round(gp, 2)

def calc_overall_diameter(specs):
    '''
    # width cut-offs ; 5 - 12.5 | 27 - 40 | 145 - 335
    # aspect ratio cut-off : 25
    # diameter issues: none
    
    >>> calc_overall_diameter('225/35/18')
    24.2
    >>> calc_overall_diameter('7.5/R/16')
    31.7
    >>> calc_overall_diameter('35/12.5/17')
    35.0
    
    1710: 12.5/80/16 -> 38.5/12.5/16
    827: 12.5/80/16 -> 36/12.5/16
    810: 12.5/80/15 -> 38/12.5/16
    804: 10.5/80/15 -> 31/10.5/15
    '''
    w, ar, d = specs.split('/')
    w = float(w)
    ar = 0.82 if ar == 'R' else float(ar)/100.0
    d = float(d)
    
    if w <= 10:
        return round((w + 0.35)*2+d, 2)
    
    elif 10 < w < 25:
        return np.NaN
    
    elif 27 <= w <= 60:
        return round(w, 1)
    
    elif w > 120:
        return round((w*ar*2)/25.4 + d, 2)
    
    else:
        return np.NaN
    
       
@st.cache_data
def get_gulong_data():
    '''
    Get gulong.ph data from backend
    
    Returns
    -------
    df : dataframe
        Gulong.ph product info dataframe
    '''
    #df = pd.read_csv('http://app.redash.licagroup.ph/api/queries/130/results.csv?api_key=JFYeyFN7WwoJbUqf8eyS0388PFE7AiG1JWa6y9Zp')
    # http://app.redash.licagroup.ph/queries/131
    url1 =  "http://app.redash.licagroup.ph/api/queries/131/results.csv?api_key=BdUhcTVmwDEqP5aYKpSolS5ApT2lig4hpdDqIPJq"

    df = pd.read_csv(url1, parse_dates = ['supplier_price_date_updated','product_price_date_updated'])
    #df_data.loc[df_data['sale_tag']==0,'promo'] =df_data.loc[df_data['sale_tag']==0,'srp']
    df = df[['product_id', 'make','model', 'section_width', 'aspect_ratio', 'rim_size' ,'pattern', 
             'load_rating','speed_rating','stock','name','cost','srp', 'promo', 'mp_price',
             'b2b_price' , 'supplier_price_date_updated','product_price_date_updated',
             'supplier_id','sale_tag', 'promo_tag']]
    # df = df[df.is_model_active==1].rename(columns={'model': 'sku_name',
    #                                                'pattern' : 'name',
    #                                                'make' : 'brand',
    #                                                'section_width':'width', 
    #                                                'rim_size':'diameter', 
    #                                                'price' : 'price_gulong'}).reset_index()
    df = df.rename(columns={'model': 'sku_name',
                            'name': 'supplier',
                            'pattern' : 'name',
                            'make' : 'brand',
                            'section_width':'width', 
                            'rim_size':'diameter', 
                            'promo' : 'price_gulong'}).reset_index(drop = True)
 
    
    
    #df.loc[:, 'raw_specs'] = df.apply(lambda x: raw_specs(x), axis=1)
    df.loc[df['sale_tag']==0, 'price_gulong'] = df.loc[df['sale_tag']==0, 'srp']
    df.loc[:, 'width'] = df.apply(lambda x: clean_width(x['width']), axis=1)
    df.loc[:, 'aspect_ratio'] = df.apply(lambda x: clean_aspect_ratio(x['aspect_ratio']), axis=1)    
    df.loc[:, 'diameter'] = df.apply(lambda x: clean_diameter(x['diameter']), axis=1)
    df.loc[:, 'raw_specs'] = df.apply(lambda x: combine_specs(x['width'], x['aspect_ratio'], x['diameter'], mode = 'SKU'), axis=1)
    df.loc[:, 'correct_specs'] = df.apply(lambda x: combine_specs(x['width'], x['aspect_ratio'], x['diameter'], mode = 'MATCH'), axis=1)
    df.loc[:, 'overall_diameter'] = df.apply(lambda x: calc_overall_diameter(x['correct_specs']), axis=1)
    df.loc[:, 'name'] = df.apply(lambda x: fix_names(x['name']), axis=1)
    df.loc[:, 'sku_name'] = df.apply(lambda x: combine_sku(str(x['brand']), 
                                                           str(x['width']),
                                                           str(x['aspect_ratio']),
                                                           str(x['diameter']),
                                                           str(x['name']), 
                                                           str(x['load_rating']), 
                                                           str(x['speed_rating'])), 
                                                           axis=1)
    df.loc[:, 'base_GP'] = (df.loc[:, 'price_gulong'] - df.loc[:, 'cost']).round(2)
    df.loc[:, 'promo_GP'] = df.apply(lambda x: promo_GP(x['price_gulong'], x['cost'], x['sale_tag'], x['promo_tag']), axis=1)
    df = df[df.name !='-']
    df.sort_values('product_price_date_updated', ascending = False, inplace = True)
    df.drop_duplicates(subset = ['product_id', 'sku_name', 'cost', 'price_gulong', 'supplier'])
    
    return df

@st.cache_data
def get_car_compatible():
    
    # http://app.redash.licagroup.ph/queries/183
    # GULONG - Car Compatible Tire Sizes
    def import_data():
        print ('Importing database data')
        url = 'http://app.redash.licagroup.ph/api/queries/183/results.csv?api_key=NWVzsgA5xGzhpW4xhslaJ5Nlx9o1ghM7P5a9PtHb'
        comp_data = pd.read_csv(url, parse_dates = ['created_at', 'updated_at'])
    
        print ('Importing makes and models list')
        makes_list = config_carmax.import_makes()
        
        print ('Cleaning data')
        comp_data.loc[:, 'car_make'] = comp_data.apply(lambda x: config_carmax.clean_makes(x['car_make'], makes_list), axis=1)
        comp_data.loc[:, 'car_model'] = comp_data.apply(lambda x: ' '.join(x['car_model'].split('-')).upper(), axis=1)
        comp_data.loc[:, 'width'] =  comp_data.apply(lambda x: clean_width(x['section_width']), axis=1)
        comp_data.loc[:, 'aspect_ratio'] = comp_data.apply(lambda x: clean_aspect_ratio(x['aspect_ratio']), axis=1)
        comp_data.loc[:, 'diameter'] = comp_data.apply(lambda x: clean_diameter(x['rim_size']), axis=1)
        return comp_data
    
    start_time = time.time()
    print ('Start car comparison tire size data import')
    comp_data = import_data()
    
    #comp_data.loc[:,'year'] = comp_data.year.astype(str)
    print ('Imported tire size car comparison data')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return comp_data

def tire_select(df_data):
    '''
    Displays retention info of selected customers.

    Parameters
    ----------
    df_data : dataframe
    df_retention : dataframe
    models : list
        list of fitted Pareto/NBD and Gamma Gamma function

    Returns
    -------
    df_retention : dataframe
        df_retention with updated values

    '''
    # Reprocess dataframe entries to be displayed
    df_merged = df_data.copy()
    
    # table settings
    df_display = df_merged.sort_values(['promo_GP', 'base_GP', 'price_gulong'])
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection('single', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gb.configure_column('sku_name', 
                        headerCheckboxSelection = True,
                        width = 400)
    gridOptions = gb.build()
    
    # selection settings
    data_selection = AgGrid(
        df_display,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=True,
        height= min(33*len(df_display), 400), 
        reload_data=False)
    
    selected = data_selection['selected_rows']
    
    if selected:           
        # row/s are selected
        
        df_selected = [df_display[df_display.sku_name == selected[checked_items]['sku_name']]
                             for checked_items in range(len(selected))]
        
        df_list = pd.concat(df_selected)
        #st.dataframe(df_list)    

    else:
        st.write('Click on an entry in the table to display customer data.')
        df_list = df_display
        
    return df_list

if __name__ == '__main__':
    
    st.title('Gulong Recommendation Model')
    
    df = get_gulong_data()
    car_comp = get_car_compatible()
    
    display_cols = ['sku_name', 'width', 'aspect_ratio', 'diameter',
                    'load_rating', 'speed_rating', 'overall_diameter', 'cost', 
                    'srp', 'price_gulong', 'mp_price', 'b2b_price', 'base_GP',
                    'promo_GP']
    
    with st.sidebar:
        st.header('Tire Selection')
        
        tab_size, tab_car = st.tabs(['By Size', 'By Car Model'])
        
        with tab_size:
            
            w_list = ['Any Width'] + list(sorted(df.width.unique()))
            
            width = st.selectbox('Width',
                                 options = w_list,
                                 index = 0)
            if width == 'Any Width':
                w_filter = df
            else:
                w_filter = df[df['width'] == width]
            
            
            ar_list = ['Any Aspect Ratio'] + list(sorted(w_filter.aspect_ratio.unique()))
            
            aspect_ratio = st.selectbox('Aspect Ratio',
                                 options = ar_list,
                                 index = 0)
            
            if aspect_ratio == 'Any Aspect Ratio':
                ar_filter = w_filter
            else:
                ar_filter = w_filter[w_filter['aspect_ratio'] == aspect_ratio]
            
            d_list = ['Any Rim Diameter'] + list(sorted(ar_filter.diameter.unique()))
            rim_diameter = st.selectbox('Rim Diameter',
                                 options = d_list,
                                 index = 0)
            
            if rim_diameter == 'Any Rim Diameter':
                final_filter = ar_filter
            else:
                final_filter = ar_filter[ar_filter['diameter'] == rim_diameter]
                
        with tab_car:
            
            make_list = ['Any make'] + list(sorted(car_comp.car_make.unique()))
            
            make = st.selectbox('Make',
                                 options = make_list,
                                 index = 0)
            if make == 'Any make':
                make_filter = car_comp
            else:
                make_filter = car_comp[car_comp['car_make'] == make]
            
            
            model_list = ['Any Model'] + list(sorted(make_filter.car_model.value_counts().keys()))
            
            model = st.selectbox('Model',
                                 options = model_list,
                                 index = 0)
            
            if model == 'Any Model':
                model_filter = make_filter
            else:
                model_filter = make_filter[make_filter['car_model'] == model]
            
            y_list = ['Any Year'] + list(sorted(model_filter.car_year.unique()))
            year = st.selectbox('Year',
                                 options = y_list,
                                 index = 0)
            
            if year == 'Any Year':
                y_filter = model_filter
            else:
                y_filter = model_filter[model_filter['car_year'] == year]

            final_filter = df[(df.width.isin(y_filter.width.unique())) & \
                (df.aspect_ratio.isin(y_filter.aspect_ratio.unique())) & \
                    (df.diameter.isin(y_filter.diameter.unique()))]
            
    # main window
    selected = final_filter.copy()
    tire_selected = tire_select(selected[display_cols])
    
    # find products with overall diameter within 3% error
    OD = tire_selected.overall_diameter.unique()[0]
    
    # calculate overall diameter % diff
    df_temp = df.copy()
    df_temp.loc[:, 'od_diff'] = df_temp.overall_diameter.apply(lambda x: round(abs((x - OD)*100/OD), 2))
    compatible = df_temp[(df_temp.od_diff <= 3) & \
                         ((df_temp.promo_GP >= tire_selected.promo_GP.max()) & \
                          (df_temp.base_GP >= tire_selected.base_GP.max()))]
    
    with st.expander('**Product Recommendation**', 
                     expanded = len(compatible)):
        
        st.info("""Recommended tires are those within ~3% change of selected tire's overall diameter.
                Resulting recommended tires are then filtered by atleast selected tire's GP,
                and then finally sorted by percent diff in overall diameter.""")
        
        if len(tire_selected) < len(selected):
            if len(compatible) == 0:
                st.error('No recommended tires found.')
            else:
                st.dataframe(compatible[display_cols + ['od_diff']].sort_values('od_diff', ascending = True))
        else:
            pass
