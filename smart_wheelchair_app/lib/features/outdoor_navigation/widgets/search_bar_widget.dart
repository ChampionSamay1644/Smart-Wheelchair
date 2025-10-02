import 'package:flutter/material.dart';

class SearchBarWidget extends StatelessWidget {
  final VoidCallback onTap;
  final String destinationName;

  const SearchBarWidget({
    super.key,
    required this.onTap,
    required this.destinationName,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 6.0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15.0)),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16.0),
        height: 60,
        child: Row(
          children: [
            const Icon(Icons.search, color: Colors.blue, size: 28),
            const SizedBox(width: 10),
            Expanded(
              child: GestureDetector(
                onTap: onTap,
                child: Container(
                  color: Colors.transparent,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        destinationName.isNotEmpty
                            ? destinationName
                            : 'Where do you want to go?',
                        style: TextStyle(
                          fontSize: 16.0,
                          color: destinationName.isNotEmpty
                              ? Colors.black
                              : Colors.grey,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      if (destinationName.isNotEmpty)
                        const Text(
                          'Tap to change destination',
                          style: TextStyle(fontSize: 12, color: Colors.grey),
                        ),
                    ],
                  ),
                ),
              ),
            ),
            if (destinationName.isNotEmpty)
              IconButton(
                icon: const Icon(Icons.clear, color: Colors.grey),
                onPressed: () {},
                tooltip: 'Clear destination',
              ),
          ],
        ),
      ),
    );
  }
}
