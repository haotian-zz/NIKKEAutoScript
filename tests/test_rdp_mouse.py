"""RDP routing checks never send input or install drivers."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


source = Path(__file__).resolve().parents[1] / 'module/device/win/virtual_mouse/rdp_mouse.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
names = {'SessionMouseError', 'button_edges', 'absolute_point', 'select_session_device'}
namespace = {}
exec(compile(ast.Module(body=[node for node in tree.body if getattr(node, 'name', None) in names],
                        type_ignores=[]), str(source), 'exec'), namespace)


class RdpRoutingLogicTests(unittest.TestCase):
    def test_only_one_observed_session_device_is_accepted(self):
        choose = namespace['select_session_device']
        self.assertEqual(choose({15}, {13, 15}), 15)
        for observed in (set(), {13, 15}, {11}):
            with self.subTest(observed=observed), self.assertRaises(OSError):
                choose(observed, {13, 15})

    def test_button_transitions_do_not_repeat_held_buttons(self):
        edge = namespace['button_edges']
        self.assertEqual([edge(0, 1), edge(1, 1), edge(1, 0)], [1, 0, 2])
        self.assertEqual(edge(1, 2), 2 | 4)
        self.assertEqual(edge(7, 0), 2 | 8 | 32)

    def test_absolute_coordinates_cover_desktop_without_clipping_invalid_targets(self):
        point = namespace['absolute_point']
        self.assertEqual(point(0, 0, (0, 0, 2560, 1440)), (0, 0))
        self.assertEqual(point(-1280, 720, (-2560, 0, 5120, 1440)), (16384, 32768))
        for target in ((2560, 0), (0, 1440), (-1, 10)):
            with self.subTest(target=target), self.assertRaises(OSError):
                point(*target, (0, 0, 2560, 1440))


if os.name == 'nt':
    from module.device.win.virtual_mouse import rdp_mouse as rdp


@unittest.skipUnless(os.name == 'nt', 'Windows session detection')
class RdpDetectionTests(unittest.TestCase):
    def test_protocol_fallback_and_console_detection(self):
        for current, protocol, expected in ((2, 2, 2), (1, 0, None)):
            def process_session(pid, pointer):
                pointer._obj.value = current
                return 1
            with self.subTest(session=current), \
                    patch.object(rdp._kernel, 'ProcessIdToSessionId', side_effect=process_session), \
                    patch.object(rdp._kernel, 'WTSGetActiveConsoleSessionId', return_value=1), \
                    patch.object(rdp._user, 'GetSystemMetrics', return_value=0), \
                    patch.object(rdp.win32ts, 'WTSQuerySessionInformation', return_value=protocol):
                self.assertEqual(rdp.remote_session_id(), expected)

    def test_unknown_nonconsole_session_does_not_fall_back_to_physical_input(self):
        from module.device.win.virtual_mouse import driver_mouse as native
        with patch.object(native, '_shared_mouse', None), \
                patch.object(rdp, 'remote_session_id', side_effect=rdp.SessionMouseError('unknown session')), \
                patch.object(native, 'VirtualMouse') as physical:
            with self.assertRaises(rdp.SessionMouseError):
                native.shared_mouse()
        physical.assert_not_called()


@unittest.skipUnless(os.name == 'nt', 'Windows input adapter')
class RdpAdapterTests(unittest.TestCase):
    def setUp(self):
        self.session = patch.object(rdp, 'remote_session_id', return_value=2)
        self.session.start()
        self.addCleanup(self.session.stop)

    def candidates(self):
        physical = Mock(hardware_id=Mock(return_value='HID\\physical'))
        foreign = Mock(index=13, hardware_id=Mock(return_value=rdp.RDP_MOUSE_ID))
        own = Mock(index=15, hardware_id=Mock(return_value=rdp.RDP_MOUSE_ID))
        def create(index):
            if index == 10:
                return physical
            if index == 13:
                return foreign
            if index == 15:
                return own
            raise OSError('No device')
        return create, physical, foreign, own

    def test_binding_retains_only_current_session_channel(self):
        create, physical, foreign, own = self.candidates()
        mouse = rdp.RdpMouseDevice()
        observed = []
        def identify(channels):
            observed.extend(channels)
            return 15
        with patch.object(rdp, '_Channel', side_effect=create), \
                patch.object(rdp, '_identify_session', side_effect=identify):
            self.assertTrue(mouse.open())
        self.assertEqual(set(observed), {13, 15})
        physical.prepare.assert_not_called()
        physical.write.assert_not_called()
        physical.close.assert_called_once()
        foreign.close.assert_called_once()
        self.assertIs(mouse.channel, own)
        mouse.close()
        own.close.assert_called_once()

    def test_ambiguous_binding_closes_all_channels_without_fallback(self):
        create, physical, foreign, own = self.candidates()
        mouse = rdp.RdpMouseDevice()
        with patch.object(rdp, '_Channel', side_effect=create), \
                patch.object(rdp, '_identify_session', side_effect=rdp.SessionMouseError('ambiguous')):
            self.assertFalse(mouse.open())
        self.assertIsNone(mouse.channel)
        for channel in (physical, foreign, own):
            channel.close.assert_called_once()

    def test_console_cannot_use_rdp_adapter(self):
        with patch.object(rdp, 'remote_session_id', return_value=None), patch.object(rdp, '_Channel') as channel:
            self.assertFalse(rdp.RdpMouseDevice().open())
        channel.assert_not_called()

    def test_session_change_prevents_all_reports(self):
        mouse = rdp.RdpMouseDevice()
        mouse.channel, mouse.session_id = Mock(), 2
        mouse.position = (100, 100)
        with patch.object(rdp, 'remote_session_id', return_value=3):
            self.assertFalse(mouse.send(buttons=1))
        mouse.channel.write.assert_not_called()

    def test_press_release_and_wheel_use_absolute_position_and_edges(self):
        mouse = rdp.RdpMouseDevice()
        mouse.channel, mouse.session_id = Mock(), 2
        desktop = {76: 0, 77: 0, 78: 2560, 79: 1440}
        with patch.object(rdp._user, 'GetSystemMetrics', side_effect=desktop.__getitem__):
            self.assertTrue(mouse.move_absolute(1280, 720))
            self.assertTrue(mouse.send(buttons=1))
            self.assertTrue(mouse.send(buttons=1))
            self.assertTrue(mouse.send(buttons=0, wheel=-1))
        packets = [call.args[0] for call in mouse.channel.write.call_args_list]
        self.assertEqual([packet.buttons for packet in packets], [0, 1, 0, 2 | 0x400])
        self.assertTrue(all(packet.flags == 3 and packet.x == 32768 for packet in packets))
        self.assertEqual(packets[-1].wheel, (-120) & 0xFFFF)

    def test_failed_binding_never_repairs_console_driver(self):
        from module.device.win.virtual_mouse import input as control
        from module.exception import RequestHumanTakeover
        handler = control.VirtualMouseInput.__new__(control.VirtualMouseInput)
        handler.mouse_driver = SimpleNamespace(session_local=True)
        handler._channel_ready = Mock(return_value=False)
        with patch.object(control, 'claim_scheme_mutex', return_value=True), \
                patch.object(control, 'repair_driver') as repair:
            with self.assertRaises(RequestHumanTakeover):
                handler._preflight()
        repair.assert_not_called()

    def test_rdp_readiness_does_not_depend_on_console_hid(self):
        from module.device.win.virtual_mouse import input as control
        handler = control.VirtualMouseInput.__new__(control.VirtualMouseInput)
        handler.mouse_driver = SimpleNamespace(session_local=True, open=lambda: True)
        with patch.object(control, 'sub_device_present', return_value=False) as physical:
            self.assertTrue(handler._channel_ready())
        physical.assert_not_called()


if __name__ == '__main__':
    unittest.main()
