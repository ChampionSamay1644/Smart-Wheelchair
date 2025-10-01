import 'package:flutter/material.dart';
import 'features/outdoor_navigation/outdoor_navigation_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Smart Wheelchair',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      routes: {
        '/': (context) => const MyHomePage(title: 'Smart Wheelchair'),
        '/outdoor-navigation': (context) => const OutdoorNavigationPage(),
      },
      initialRoute: '/',
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  // Helper method to build consistent feature buttons
  Widget _buildFeatureButton(
    BuildContext context, 
    String title, 
    IconData icon, 
    VoidCallback onPressed,
  ) {
    return SizedBox(
      width: 300,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.all(20),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            Icon(icon, size: 36, color: Theme.of(context).primaryColor),
            const SizedBox(width: 16),
            Text(
              title,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'Smart Wheelchair Features',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 40),
            _buildFeatureButton(
              context, 
              'Outdoor Navigation', 
              Icons.map, 
              () => Navigator.pushNamed(context, '/outdoor-navigation'),
            ),
            const SizedBox(height: 16),
            _buildFeatureButton(
              context, 
              'Indoor Navigation', 
              Icons.home, 
              () {},
            ),
            const SizedBox(height: 16),
            _buildFeatureButton(
              context, 
              'Health Monitoring', 
              Icons.favorite, 
              () {},
            ),
          ],
        ),
      ),
    );
  }
}
}
