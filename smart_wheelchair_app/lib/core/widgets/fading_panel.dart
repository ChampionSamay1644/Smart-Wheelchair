import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/notifications_provider.dart';

/// A widget that can be shown with a fade transition for overlays, notifications,
/// errors, etc.
class FadingPanel extends StatefulWidget {
  /// The content to display in the panel.
  final Widget child;

  /// The duration of the fade animation. Defaults to 300ms.
  final Duration duration;

  const FadingPanel({
    super.key,
    required this.child,
    this.duration = const Duration(milliseconds: 300),
  });

  @override
  State<FadingPanel> createState() => _FadingPanelState();
}

class _FadingPanelState extends State<FadingPanel>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _opacity;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(duration: widget.duration, vsync: this);

    _opacity = CurvedAnimation(parent: _controller, curve: Curves.easeInOut);

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(opacity: _opacity, child: widget.child);
  }
}

/// A small non-blocking fading notification panel that appears at the top-right
/// of the screen for a short duration when a new notification event is added
/// to the [NotificationsProvider].
class NotificationPanel extends StatefulWidget {
  /// How long the panel remains visible
  final Duration visibleDuration;

  /// How long the fade animation takes
  final Duration fadeDuration;

  const NotificationPanel({
    super.key,
    this.visibleDuration = const Duration(seconds: 3),
    this.fadeDuration = const Duration(milliseconds: 300),
  });

  @override
  State<NotificationPanel> createState() => _NotificationPanelState();
}

class _NotificationPanelState extends State<NotificationPanel>
    with SingleTickerProviderStateMixin {
  OverlayEntry? _overlayEntry;
  NotificationEvent? _shownEvent;
  bool _isShowing = false;

  @override
  void initState() {
    super.initState();
    // Use a post-frame callback so that context/overlay is available
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final provider = Provider.of<NotificationsProvider>(
        context,
        listen: false,
      );
      provider.addListener(_onNotificationsChanged);
    });
  }

  @override
  void dispose() {
    final provider = Provider.of<NotificationsProvider>(context, listen: false);
    provider.removeListener(_onNotificationsChanged);
    _removeOverlay();
    super.dispose();
  }

  void _onNotificationsChanged() {
    final provider = Provider.of<NotificationsProvider>(context, listen: false);
    final event = provider.lastEvent;
    if (event == null) return;

    // If event is the same as currently shown, ignore
    if (_isShowing && _shownEvent == event) return;

    _shownEvent = event;
    _showOverlay(event);
  }

  void _showOverlay(NotificationEvent event) async {
    _removeOverlay();

    _isShowing = true;
    final overlay = Overlay.of(context, rootOverlay: true);

    final entry = OverlayEntry(
      builder: (context) {
        return _PanelWidget(event: event, fadeDuration: widget.fadeDuration);
      },
    );

    _overlayEntry = entry;
    overlay.insert(entry);

    // Keep visible for the requested duration, then fade out and remove
    await Future.delayed(widget.visibleDuration + widget.fadeDuration);
    _removeOverlay();
  }

  void _removeOverlay() {
    if (_overlayEntry != null) {
      try {
        _overlayEntry!.remove();
      } catch (_) {}
      _overlayEntry = null;
    }
    _isShowing = false;
    _shownEvent = null;
  }

  @override
  Widget build(BuildContext context) {
    // This widget doesn't render anything by itself. It simply hooks into the
    // NotificationsProvider and shows overlays on top of the app.
    return const SizedBox.shrink();
  }
}

class _PanelWidget extends StatefulWidget {
  final NotificationEvent event;
  final Duration fadeDuration;

  const _PanelWidget({required this.event, required this.fadeDuration});

  @override
  __PanelWidgetState createState() => __PanelWidgetState();
}

class __PanelWidgetState extends State<_PanelWidget>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _opacity;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.fadeDuration,
    );
    _opacity = CurvedAnimation(parent: _controller, curve: Curves.easeInOut);

    // Start with fade in
    _controller.forward();

    // After visible duration - fade out
    Future.delayed(Duration.zero, () async {
      // Wait the visible portion (visible duration accounted by outer widget)
      await Future.delayed(Duration.zero);
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final safeTop = MediaQuery.of(context).padding.top + 12.0;
    return Positioned(
      top: safeTop,
      right: 12.0,
      child: FadeTransition(
        opacity: _opacity,
        child: Material(
          elevation: 6,
          borderRadius: BorderRadius.circular(8),
          color: Colors.transparent,
          child: Container(
            constraints: const BoxConstraints(maxWidth: 360),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              borderRadius: BorderRadius.circular(8),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withAlpha(31),
                  blurRadius: 8,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.notifications, size: 20),
                const SizedBox(width: 10),
                Flexible(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        widget.event.title,
                        style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        widget.event.body,
                        style: Theme.of(context).textTheme.bodyMedium,
                        maxLines: 3,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
