'''
Author: Shane Motley
Creation Date: 2020.02.08
Purpose: Create two (2) graphs in both HTML and png form from Lake Spaulding Data.
        (1) A stacked bar graph with one bar being a frame of the average precip in a year
             and a second blue bar showing the total for this water year.
        (2) A cumulative line graph showing this year's total-to-date, last wy, wettest wy
             and driest wy, along with a filled area showing the average.
Method: All data obtained from CDEC, with two data pulls occurring on every run.
        (1) Daily precip data for the current water year.
        (2) Historical monthly data for the last 100 years.
'''


from plotly import graph_objs as go
from calendar import monthrange
from plotly.subplots import make_subplots
import requests
import pathlib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from PIL import Image
import platform


def create_plots():
    '''
    Perform the two data pulls from CDEC.
    Creates the following dataframes:
        df:             CDEC daily precip data for current water year.
        df_all_years:   Monthly precip data for the last 100 years.
    OUTPUT: Two html files and two static images.
    :return:
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    wy = datetime.today().year              # Water Year
    month_num = datetime.today().month + 2  # Current month number

    # The "AS-OF" date. Date of data pull
    data_pull_date = (datetime.today() - timedelta(days=1)).strftime("%b %d")

    # A date range spanning every day of the entire water year.
    idx = pd.date_range(f'10.01.{wy-1}', f'9.30.{wy}')
    if datetime.today().month >= 10:    # The water year will be year + 1 if it's Oct, Nov, or Dec
        wy = wy + 1
        month_num = month_num - 12
        idx = pd.date_range(f'10.01.{wy - 1}', f'9.30.{wy}')
    try:
        # Daily Data Pull
        response = requests.get(
            url="https://cdec.water.ca.gov/dynamicapp/QueryDaily?s=LSP",
            params={"end": f"{wy}-10-1",
                    "span": "1year"
                    },
        )
        print('Response HTTP Status Code: {status_code}'.format(status_code=response.status_code))

        # Historical Data Pull
        all_years = requests.get (
            url="https://cdec.water.ca.gov/dynamicapp/QueryWY",
            params={"Stations": "LSP",
                    "span": "100+years",
                    "SensorNums":2
                    },
        )
        print('Response HTTP Status Code: {status_code}'.format(status_code=all_years.status_code))

        # Current data to-date
        df = pd.read_html(response.content)[0]

        # For some reason, the dataframe extends to 10-01 of the following water year instead of 9-30, so
        # we need to remove the last row
        df = df[:-1]

        df_driest = pd.read_csv(os.path.join(dir_path,'LSP_WY1976.csv'))
        df_wettest = pd.read_csv(os.path.join(dir_path,'LSP_WY2017.csv'))
        try:
            df_lastyear = pd.read_csv(f'LSP_WY{wy-1}.csv')
        except FileNotFoundError:
            get_last_years_data(wy)
            df_lastyear = pd.read_csv(f'LSP_WY{wy-1}.csv')

        # Monthly Data
        df_all_years = pd.read_html(all_years.content)[0]

        # Missing data in CDEC is shown as a string "--". Change this.
        df_all_years = df_all_years.apply(pd.to_numeric, errors='coerce')

        # Remove the total column so that we can loop through months.
        df_all_years.drop('WY Total', axis=1, inplace=True)

        # Construct an empty dataframe for the average values with every day of the year.
        df_ave = pd.DataFrame(index=idx, columns=["Average"])

        # Holds value of ave precip on last day of month.
        cumulative = 0

        # Fill each row of empty dataframe
        for index, row in df_ave.iterrows():
            # Month abbreviation
            month_str = index.strftime("%b")
            days_in_month = monthrange(index.year, index.month)[1]

            # The total average precipitation for the given month.
            month_ave = df_all_years[month_str].mean()

            # Each month has a new average per day, but the first day of each month needs to start at the last value
            # from the previous month.
            if index.day == 1 and index.month !=10:
                # Current row number
                row_num = df_ave.index.get_loc(index)

                # The value from the previous row.
                cumulative = df_ave["Average"].iloc[row_num-1]

            # Daily average is the interpolation amount for the day + the amount from the previous day
            day_ave = (month_ave / days_in_month) * index.day + cumulative

            # Write value to data frame
            df_ave.loc[[index], ["Average"]] = day_ave

        # CDEC places NA values as '--'
        df.replace('--', np.nan, inplace=True)

        # Change the name of the first column
        df.rename(columns={df.columns[0]:"DATE_TIME"}, inplace=True)

        # Drop "Unnamed" columns
        cols = [c for c in df.columns if c.lower()[:7] != "unnamed"]
        df = df[cols]

        # Make sure all columns except the Date column are numeric instead of string
        df[cols[1:]] = df[cols[1:]].apply(pd.to_numeric)

        # Convert Date
        df['DATE_TIME'] = pd.to_datetime(df["DATE_TIME"])

        # Monthly values for the bar chart will be the daily values grouped by month.
        df_monthly = df.groupby(pd.Grouper(key="DATE_TIME", freq="1M")).sum()

        historical_average = []
        # Get the average values from the historical data spanning back 50 years.
        for column in df_all_years.columns[1:]:
            historical_average.append(round(df_all_years[column].mean(),1))

        # Create new column and insert the historical average data into the column.
        df_monthly["Average"] = historical_average
        # Main bar chart
        barChart=(go.Figure(data=[go.Bar(
                                    x=(df_monthly.index).strftime("%b"),
                                    y=df_monthly['PPT INC INCHES'],
                                    marker_line_width=2,
                                    name=f"{wy} Water Year",
                                    hovertemplate="Observed: %{y} inches<extra></extra>",
                                    marker_line_color='rgb(0,0,0)',
                                    marker={"color":'rgb(71,165,252)'},
                                    texttemplate='%{y:.1f}',
                                    textposition="auto",
                                    textfont=dict(
                                        family="Arial",
                                        size=24,
                                        color="rgb(0, 0, 153)"
                                    )
                                    ),
                              go.Bar(
                                    x=(df_monthly.index).strftime("%b"),
                                    y=df_monthly['Average'],
                                    name="50 Year Average",
                                    hovertext=df_monthly['Average'],
                                    hovertemplate="Average: %{y} inches<extra></extra>",
                                    marker_color='rgba(255,255,255,0)',
                                    marker_line_width=2,
                                    marker_line_color='rgb(0,0,0)',
                                      texttemplate='%{hovertext}',
                                      textposition="auto",
                                      textfont=dict(
                                          family="Arial",
                                          size=24,
                                          color="rgba(0, 0, 0)"
                                      )
                            )
                           ])
            )
        barChart.update_layout(barmode='overlay',
                          paper_bgcolor='rgb(255,255,255)',
                          plot_bgcolor="rgb(255,255,255)",
                          title=f"Lake Spaulding Precipitation: Water Year {wy}",
                          showlegend=True,
                          hovermode="x unified",
                          legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="right",
                                    x=1),
                          font=dict(
                            size=24,
                              color="black"
                            )
                          )

        # The box that will have the "AS-OF" information.
        text_box_x = (month_num+1)
        text_box_y = 13                 # This will ensure it's above the "normal" bars
        barChart.add_annotation(x=month_num, y=df_monthly['PPT INC INCHES'].iloc[month_num],
                           text=f"Total Through {data_pull_date}",
                           showarrow=True,
                           ax=text_box_x,         # Move box over 1 space from the month number
                           ay=text_box_y,         # Put box above all the normal bars
                           axref="x",             # Make sure the reference to the annotation is the x axis (not pixels)
                           ayref="y",             # Make sure the reference to the annotation is the y axis (not pixels)
                           arrowcolor="black",
                           arrowsize=1,
                           arrowwidth=3,
                           arrowhead=1,
                           bordercolor="rgb(155, 155, 155)",
                           borderwidth=2,
                           borderpad=4,
                           bgcolor="rgb(158, 225, 247)",
                           opacity=0.7
                           )
        # Place tick marks on the graph and put a boarder around the entire graph (mirror).
        barChart.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True,
                         ticks="outside", tickwidth=2, tickcolor='black', ticklen=10)
        barChart.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True,
                         ticks="outside", tickwidth=2, tickcolor='black', ticklen=10)

        # Line chart info.
        lineChart = (go.Figure(data=[go.Scatter(
                                    x=(df_ave.index),
                                    y=df_ave['Average'],
                                    name="50 Year Average",
                                    hovertext=df_ave['Average'],
                                    hovertemplate="Average: %{y:.1f}\"<extra></extra>",
                                    fill='tozeroy',
                                    line=dict(width=0.5, color='rgb(111, 231, 219)'),
                                    marker_line_width=2,
                                      texttemplate='%{hovertext}',
                                      textposition="top center",
                                      textfont=dict(
                                          family="Arial",
                                          size=24,
                                          color="rgba(0, 0, 0)"
                                      )
                                ),
                                go.Scatter(
                                    x=(df["DATE_TIME"]),
                                    y=df['RAIN INCHES'],
                                    marker_line_width=2,
                                    name=f"{wy} Water Year",
                                    hovertemplate="Observed: %{y}\"<extra></extra>",
                                    line=dict(width=4, color='rgb(71,165,252)',),
                                    texttemplate='%{y:.1f}',
                                    textposition="top center",
                                    textfont=dict(
                                        family="Arial",
                                        size=24,
                                        color="rgb(0, 0, 153)"
                                    )
                                ),
                            ]
                        )
                    )

        # Wettest year trace
        lineChart.add_trace(go.Scatter(x=df_ave.index,
                                  y=df_wettest['PRECIP'],
                                  hovertemplate="Wettest: %{y}\"<extra></extra>",
                                  mode='lines',
                                  name='Wettest (WY 2017)'))
        # Driest year trace
        lineChart.add_trace(go.Scatter(x=df_ave.index,
                                  y=df_driest['PRECIP'],
                                  hovertemplate="Driest: %{y}\"<extra></extra>",
                                  mode="lines",
                                  name='Driest (WY 1977)'))
        # Last Year trace
        lineChart.add_trace(go.Scatter(x=df_ave.index,
                                  y=df_lastyear['PRECIP'],
                                  mode='lines',
                                  hovertemplate="Last Year: %{y}\"<extra></extra>",
                                  name=f'Last Year (WY {wy-1})'))

        # Update all aspects of the chart
        lineChart.update_layout(paper_bgcolor='rgb(255,255,255)',
                          plot_bgcolor="rgb(255,255,255)",
                          title=f"Lake Spaulding Precipitation: Water Year {wy}",
                          showlegend=True,
                          hovermode="x unified",
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0),
                          font=dict(
                              size=24,
                              color="black"
                          )
                          )

        # For the annotation, we need to know what row the latest data are in so we can get the
        # that info into the "as-of" box. NAN's are in all other dates through the end of the water year,
        # so you can't simply get the last value in the column
        today_date_row = df[df['DATE_TIME'] == datetime.today().strftime("%Y-%m-%d 00:00:00")].index[0]
        ave_date_row = next(iter(np.where(df_ave.index == datetime.now().replace(hour=0,minute=0,second=0,microsecond=0))[0]), 'not matched')

        # At 6:00 am, when this automatically runs,  the cumulative precip is not avail (nan). So to get this value,
        # and plot it, add the value from the day before (day before yesterday) and add the daily value from yesterday.
        if pd.isnull(df['RAIN INCHES'].iloc[today_date_row-1]):
            # df['RAIN INCHES'].iloc[today_date_row - 1]= df['RAIN INCHES'].iloc[today_date_row-2] + \
            #                                             df['PPT INC INCHES'].iloc[today_date_row-1]
            df.at[(today_date_row - 1), 'RAIN INCHES'] = df['RAIN INCHES'].iloc[today_date_row - 2] + \
                                                         df['PPT INC INCHES'].iloc[today_date_row - 1]

        lineChart.add_annotation(x=datetime.today()-timedelta(days=1), y=df['RAIN INCHES'].iloc[today_date_row-1],
                          text=f"Through {data_pull_date}<br>"
                               f"Total = {df['RAIN INCHES'].max()} <br>"
                               f"% Ave = {int(100*(df['RAIN INCHES'].iloc[today_date_row-1]/df_ave['Average'].iloc[ave_date_row-1]))}%",
                          showarrow=True,
                          ax=100,
                          ay=-100,
                          arrowcolor="black",
                          arrowsize=1,
                          arrowwidth=3,
                          arrowhead=1,
                          bordercolor="rgb(155, 155, 155)",
                          borderwidth=2,
                          borderpad=4,
                          bgcolor="rgb(158, 225, 247)",
                          opacity=0.7
                          )

        lineChart.update_xaxes(showline=True, linewidth=2, linecolor='black',
                         ticks="outside", tickwidth=2, tickcolor='black', ticklen=10)
        lineChart.update_yaxes(showline=True, linewidth=2, linecolor='black',
                         ticks="outside", tickwidth=2, tickcolor='black', ticklen=10,
                          range=[0, df_wettest['PRECIP'].max()+5])

        wfolder_path = os.path.join(os.path.sep, 'home', 'smotley', 'images', 'weather_email')
        if platform.system() == "Windows":
            wfolder_path = pathlib.Path('G:/','Energy Marketing','Weather','Programs','Lake_Spaulding')

        barChart.write_html(os.path.join(wfolder_path, 'LSP_Bar_WY2021.html'),include_plotlyjs='cdn',
                            include_mathjax='cdn', full_html=False)
        lineChart.write_html(os.path.join(wfolder_path, 'LSP_Line_WY2021.html'),include_plotlyjs='cdn',
                             include_mathjax='cdn', full_html=False)

        barChart_path = os.path.join(wfolder_path, "LSP_Bar.png")
        lineChart_path = os.path.join(wfolder_path, "LSP_Line.png")

        barChart.write_image(os.path.join(wfolder_path, "LSP_Bar.png"), width=1200, height=750, engine='kaleido')
        lineChart.write_image(os.path.join(wfolder_path, "LSP_Line.png"), width=1200, height=750, engine='kaleido')

        merge_pngs(barChart_path, lineChart_path, wfolder_path)
        return
    except requests.exceptions.RequestException:
        print('HTTP Request failed')
        return None


def merge_pngs(img1, img2, wfolder_path):
    images = [Image.open(image) for image in [img1, img2]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (int(total_width/2), total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]

    new_im.save(os.path.join(wfolder_path, 'LSP_Graphs.png'))


def get_last_years_data(wy):
    '''
    On the very first date of the water year, a csv file from the previous water year will not
    exist on the local drive. So go grab the data and download it. (Should only have to do this
    once a year).
    :param wy: current water year.
    :return:
    '''
    try:
        response = requests.get(
            url="https://cdec.water.ca.gov/dynamicapp/QueryDaily?s=LSP",
            params={"end": f"{wy-1}-10-1",
                    "span": "1year"
                    },
        )
        # Create a dataframe with two columns: (1) DATE (2) PRECIP
        df_ly = pd.read_html(response.content)[0]
        df_ly.replace('--', np.nan, inplace=True)
        df_ly.rename(columns={df_ly.columns[0]: "DATE", df_ly.columns[1]: "DAILY_PRECIP",
                              df_ly.columns[3]: "PRECIP"}, inplace=True)

        # Keep only the DATE and PRECIP columns
        df_ly = df_ly.filter(['DATE', 'PRECIP'])
        df_ly.to_csv(f'LSP_WY{wy-1}.csv')
        return

    except requests.exceptions.RequestException:
        print('HTTP Request Four Previous Water Year failed')
        return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_plots()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
