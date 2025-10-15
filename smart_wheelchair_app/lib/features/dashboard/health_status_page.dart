import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class HealthStatusPage extends StatefulWidget {
  const HealthStatusPage({super.key});

  @override
  State<HealthStatusPage> createState() => _HealthStatusPageState();
}

class _HealthStatusPageState extends State<HealthStatusPage> {
  int _selectedTimeRange = 7; // Default to 7 days

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Health Status')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildTimeRangeSelector(),
            const SizedBox(height: 24),
            _buildVitalCard(
              'Heart Rate',
              '72',
              'BPM',
              FontAwesomeIcons.heartPulse,
              Colors.red,
              _buildHeartRateChart(),
            ),
            const SizedBox(height: 16),
            _buildVitalCard(
              'Blood Pressure',
              '120/80',
              'mmHg',
              FontAwesomeIcons.heartCircleCheck,
              Colors.purple,
              _buildBloodPressureChart(),
            ),
            const SizedBox(height: 16),
            _buildVitalCard(
              'Body Temperature',
              '37.2',
              '°C',
              FontAwesomeIcons.temperatureHalf,
              Colors.orange,
              _buildTemperatureChart(),
            ),
            const SizedBox(height: 16),
            _buildVitalCard(
              'Oxygen Saturation',
              '98',
              '%',
              FontAwesomeIcons.lungs,
              Colors.blue,
              _buildOxygenChart(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTimeRangeSelector() {
    return SegmentedButton<int>(
      segments: const [
        ButtonSegment<int>(value: 1, label: Text('24h')),
        ButtonSegment<int>(value: 7, label: Text('7d')),
        ButtonSegment<int>(value: 30, label: Text('30d')),
        ButtonSegment<int>(value: 90, label: Text('90d')),
      ],
      selected: {_selectedTimeRange},
      onSelectionChanged: (Set<int> newSelection) {
        setState(() {
          _selectedTimeRange = newSelection.first;
        });
      },
    );
  }

  Widget _buildVitalCard(
    String title,
    String value,
    String unit,
    IconData icon,
    Color color,
    Widget chart,
  ) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                FaIcon(icon, color: color, size: 24),
                const SizedBox(width: 8),
                Text(title, style: Theme.of(context).textTheme.titleMedium),
                const Spacer(),
                Text(
                  value,
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: color,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 4),
                Text(unit, style: Theme.of(context).textTheme.bodyMedium),
              ],
            ),
            const SizedBox(height: 16),
            SizedBox(height: 200, child: chart),
          ],
        ),
      ),
    );
  }

  Widget _buildHeartRateChart() {
    return LineChart(
      LineChartData(
        gridData: _defaultGridData,
        titlesData: _defaultTitlesData,
        borderData: _defaultBorderData,
        lineBarsData: [
          LineChartBarData(
            spots: [
              const FlSpot(0, 70),
              const FlSpot(1, 72),
              const FlSpot(2, 75),
              const FlSpot(3, 74),
              const FlSpot(4, 71),
              const FlSpot(5, 73),
              const FlSpot(6, 72),
            ],
            isCurved: true,
            color: Colors.red,
            barWidth: 3,
            dotData: _defaultDotData,
          ),
        ],
      ),
    );
  }

  Widget _buildBloodPressureChart() {
    return LineChart(
      LineChartData(
        gridData: _defaultGridData,
        titlesData: _defaultTitlesData,
        borderData: _defaultBorderData,
        lineBarsData: [
          LineChartBarData(
            spots: [
              const FlSpot(0, 120),
              const FlSpot(1, 118),
              const FlSpot(2, 122),
              const FlSpot(3, 119),
              const FlSpot(4, 121),
              const FlSpot(5, 120),
              const FlSpot(6, 120),
            ],
            isCurved: true,
            color: Colors.purple,
            barWidth: 3,
            dotData: _defaultDotData,
          ),
        ],
      ),
    );
  }

  Widget _buildTemperatureChart() {
    return LineChart(
      LineChartData(
        gridData: _defaultGridData,
        titlesData: _defaultTitlesData,
        borderData: _defaultBorderData,
        lineBarsData: [
          LineChartBarData(
            spots: [
              const FlSpot(0, 37.0),
              const FlSpot(1, 37.1),
              const FlSpot(2, 37.2),
              const FlSpot(3, 37.1),
              const FlSpot(4, 37.2),
              const FlSpot(5, 37.3),
              const FlSpot(6, 37.2),
            ],
            isCurved: true,
            color: Colors.orange,
            barWidth: 3,
            dotData: _defaultDotData,
          ),
        ],
      ),
    );
  }

  Widget _buildOxygenChart() {
    return LineChart(
      LineChartData(
        gridData: _defaultGridData,
        titlesData: _defaultTitlesData,
        borderData: _defaultBorderData,
        lineBarsData: [
          LineChartBarData(
            spots: [
              const FlSpot(0, 98),
              const FlSpot(1, 97),
              const FlSpot(2, 98),
              const FlSpot(3, 98),
              const FlSpot(4, 99),
              const FlSpot(5, 98),
              const FlSpot(6, 98),
            ],
            isCurved: true,
            color: Colors.blue,
            barWidth: 3,
            dotData: _defaultDotData,
          ),
        ],
      ),
    );
  }

  FlGridData get _defaultGridData =>
      const FlGridData(show: true, drawVerticalLine: false);

  FlTitlesData get _defaultTitlesData => const FlTitlesData(
    show: true,
    rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
    topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
  );

  FlBorderData get _defaultBorderData => FlBorderData(
    show: true,
    border: Border(
      bottom: BorderSide(color: Colors.grey.shade300),
      left: BorderSide(color: Colors.grey.shade300),
    ),
  );

  FlDotData get _defaultDotData =>
      const FlDotData(show: true, getDotPainter: _getDefaultDotPainter);

  static FlDotCirclePainter _getDefaultDotPainter(
    FlSpot spot,
    double xPercentage,
    LineChartBarData bar,
    int index,
  ) {
    return FlDotCirclePainter(
      radius: 4,
      color: bar.color ?? Colors.black,
      strokeWidth: 2,
      strokeColor: Colors.white,
    );
  }
}
