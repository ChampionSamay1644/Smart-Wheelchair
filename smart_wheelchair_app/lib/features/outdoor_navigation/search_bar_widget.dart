import 'package:flutter/material.dart';

class SearchBarWidget extends StatefulWidget {
  final ValueChanged<String> onSearch;
  const SearchBarWidget({super.key, required this.onSearch});

  @override
  State<SearchBarWidget> createState() => _SearchBarWidgetState();
}

class _SearchBarWidgetState extends State<SearchBarWidget> {
  final TextEditingController _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              decoration: const InputDecoration(
                hintText: 'Enter destination',
                border: OutlineInputBorder(),
              ),
              onSubmitted: widget.onSearch,
            ),
          ),
          const SizedBox(width: 8),
          ElevatedButton(
            onPressed: () => widget.onSearch(_controller.text),
            child: const Text('Go'),
          ),
        ],
      ),
    );
  }
}
