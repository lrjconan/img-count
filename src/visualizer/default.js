Array.min = function( array ){
  return Math.min.apply( Math, array );
};
Array.max = function( array ){
  return Math.max.apply( Math, array );
};

var allCharts = {};

// Choose between step and time.
var xKey = "step";
var xKeyFormat = "";
if (xKey === "step") {
  xKeyFormat = ",d";
} else {
  xKeyFormat = "";
}

var timeOpt = "absolute"

// Parse CSV data.
var parseData = function(csvData) {
  // Look up the data key.
  var yKey = "";
  for (var key in csvData[0]) {
    if (key !== "step" && key !== "time") {
      yKey = key;
      break;
    }
  }

  // Assemble array into x, y tuples.
  var displayValues = csvData.map(function(item) {
    return {
      "x": item[xKey],
      "y": item[yKey]
    }
  });

  // Assemble data.
  var data = [{
                values: displayValues,
                key: yKey
              }];
  return data;
};

// Add a chart.
var addChart = function(filename) {
  nv.addGraph(function() {
      // Load data
      d3.csv(filename, function(error, csvData) {

        var data = parseData(csvData);

        // Extract y value range.
        var yValues = data[0].values.map(function(item) {return item.y});

        // Initialize chart.
        var chart = nv.models.lineChart()
            .options({
                transitionDuration: 300,
                useInteractiveGuideline: true
            })
            .yDomain([Array.min(yValues), Array.max(yValues)]);
        chart.xAxis
            .axisLabel(xKey)
            .tickFormat(d3.format(xKeyFormat));
        chart.yAxis
            .axisLabel("")
            .tickFormat(function(d) {
                if (d == null) {
                    return "N/A";
                }
                return d3.format(",.2f")(d);
            });
        d3.select("#chart1").append("svg")
            .datum(data)
            .call(chart);

        allCharts[filename] = chart;
        // nv.utils.windowResize(chart.update);
      });
  });
};

$(function(){
  addChart('train_ce.log');
});
