import 'package:flutter/material.dart';
import '../../core/localization.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:provider/provider.dart';
import '../../core/providers/api_provider.dart';
import 'dart:math' as math;

class HealthStatusPage extends StatefulWidget {
  const HealthStatusPage({super.key});

  @override
  State<HealthStatusPage> createState() => _HealthStatusPageState();
}

class _HealthStatusPageState extends State<HealthStatusPage> {
  int _selectedTimeRange = 7; // Default to 7 days

  @override
  Widget build(BuildContext context) {
    return Consumer<ApiProvider>(
      builder: (context, apiProvider, _) {
        final data = apiProvider.latestSensorData;
        final dht = data['dht11'] ?? {};
        final max30 = data['max30100'] ?? {};
        final killswitch = apiProvider.killswitchEnabled;

        return Scaffold(
          appBar: AppBar(
            title: Text(tr(context, 'health_status')),
            actions: [
              if (apiProvider.isPolling)
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16),
                  child: Icon(Icons.sync, color: Colors.green, size: 16),
                ),
            ],
          ),
          body: killswitch 
            ? _buildKillswitchMessage()
            : SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (apiProvider.selectedDeviceId == null)
                      _buildNoDeviceWarning(),
                    _buildTimeRangeSelector(),
                    const SizedBox(height: 24),
                    _buildVitalCard(
                      tr(context, 'heart_rate'),
                      max30['ir'] != null ? '${(max30['ir'] / 100).round()}' : '--',
                      'BPM',
                      FontAwesomeIcons.heartPulse,
                      Colors.red,
                      _buildHeartRateChart(),
                    ),
                    const SizedBox(height: 16),
                    _buildVitalCard(
                      tr(context, 'body_temperature'),
                      dht['temperature']?.toString() ?? '--',
                      '°C',
                      FontAwesomeIcons.temperatureHalf,
                      Colors.orange,
                      _buildTemperatureChart(),
                    ),
                    const SizedBox(height: 16),
                    _buildVitalCard(
                      'Humidity',
                      dht['humidity']?.toString() ?? '--',
                      '%',
                      FontAwesomeIcons.droplet,
                      Colors.blue,
                      _buildHumidityChart(),
                    ),
                  ],
                ),
              ),
        );
      },
    );
  }

  Widget _buildKillswitchMessage() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.warning_amber_rounded, size: 64, color: Colors.red),
          const SizedBox(height: 16),
          Text(
            'System Temporarily Disabled',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const Padding(
            padding: EdgeInsets.all(24.0),
            child: Text(
              'Hardware communication is currently disabled by the administrator. Please try again later.',
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNoDeviceWarning() {
    return Card(
      color: Colors.orange.shade100,
      child: ListTile(
        leading: const Icon(Icons.warning, color: Colors.orange),
        title: const Text('No Wheelchair Selected'),
        subtitle: const Text('Go to Settings to select your wheelchair device.'),
        trailing: TextButton(
          onPressed: () => Navigator.pushNamed(context, '/settings'),
          child: const Text('Settings'),
        ),
      ),
    );
  }

  Widget _buildHumidityChart() {
    return _buildTemperatureChart(); // Use temperature chart as placeholder for historical humidity
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
